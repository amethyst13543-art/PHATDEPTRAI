from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import logging
import math
import os
import re
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageColor, ImageFilter

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("saas-image-engine")

# ---------- Rate Limiter ----------
class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def is_allowed(self, ip: str) -> bool:
        async with self.lock:
            now = time.time()
            self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
            if len(self.requests[ip]) >= self.max_requests:
                return False
            self.requests[ip].append(now)
            return True

rate_limiter = RateLimiter(max_requests=10, window=60)

# ---------- Model Singleton ----------
_rmbg_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rmbg_session
    try:
        from rembg import new_session
        os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"
        _rmbg_session = new_session("u2netp")
        logger.info("✅ Đã tải model tách nền (u2netp).")
    except Exception as e:
        logger.critical(f"❌ Không thể tải model: {e}")
        raise RuntimeError("Khởi tạo model thất bại.") from e
    yield
    if _rmbg_session:
        del _rmbg_session
    gc.collect()

# ---------- FastAPI App ----------
app = FastAPI(title="AI Image SaaS Pro Max Ultra", version="10.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": True, "message": exc.detail})
    logger.error(f"Ngoại lệ: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error": True, "message": "Lỗi máy chủ nội bộ."})

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method == "POST":
        client_ip = request.client.host if request.client else "127.0.0.1"
        if not await rate_limiter.is_allowed(client_ip):
            return JSONResponse(status_code=429, content={"error": True, "message": "IP của bạn thao tác quá nhanh. Vui lòng thử lại sau 1 phút."})
    response = await call_next(request)
    return response

# ---------- Helpers ----------
def validate_hex_color(color: str) -> str:
    if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color):
        raise ValueError(f"Mã màu không hợp lệ: {color}")
    return color.upper()

async def read_and_validate_file(file: UploadFile, max_mb: int = 10) -> Tuple[bytes, str]:
    allowed = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    if file.content_type not in allowed:
        raise HTTPException(415, "Chỉ chấp nhận ảnh JPEG, PNG hoặc WebP.")
    content = await file.read()
    if len(content) / (1024*1024) > max_mb:
        raise HTTPException(413, f"Ảnh vượt quá {max_mb}MB.")
    return content, file.content_type

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt in ("PNG", "WEBP") and img.mode in ("RGBA", "LA", "P"):
        img.save(buf, format=fmt, lossless=True if fmt == "WEBP" else None)
    else:
        if fmt == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def limit_output_pixels(w: int, h: int, max_mp: int = 16):
    if w * h > max_mp * 1_000_000:
        raise HTTPException(413, f"Ảnh sau xử lý vượt quá {max_mp}MP. Vui lòng giảm mức phóng to hoặc dùng ảnh nhỏ hơn.")

# ---------- Chunk Processing cho Siêu phân giải ----------
def _super_resolution_chunked(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    limit_output_pixels(new_w, new_h)

    if w * h <= 2_000_000:
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return img

    dst = Image.new("RGB", (new_w, new_h))
    chunk_w = math.ceil(w / 2)
    chunk_h = math.ceil(h / 2)

    for i in range(2):
        for j in range(2):
            left = i * chunk_w
            upper = j * chunk_h
            right = min(left + chunk_w, w)
            lower = min(upper + chunk_h, h)

            tile = img.crop((left, upper, right, lower))
            tile_w, tile_h = tile.size
            new_tile_w = int(tile_w * scale)
            new_tile_h = int(tile_h * scale)
            tile_resized = tile.resize((new_tile_w, new_tile_h), Image.LANCZOS)
            tile_resized = tile_resized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            dst.paste(tile_resized, (int(left * scale), int(upper * scale)))
            del tile, tile_resized
            gc.collect()
    return dst

# ---------- Các hàm xử lý ảnh ----------
def _remove_bg(img: Image.Image, output_type: str = "transparent", hex_color: Optional[str] = None, bg_img: Optional[Image.Image] = None) -> Image.Image:
    from rembg import remove
    w, h = img.size
    if w * h > 12_000_000:
        raise ValueError("Ảnh quá lớn (>12MP).")
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    output = remove(img, session=_rmbg_session)
    del img; gc.collect()

    if output_type == "transparent":
        return output
    elif output_type == "hex_color":
        if not hex_color: raise ValueError("Thiếu mã màu.")
        validate_hex_color(hex_color)
        rgb = ImageColor.getrgb(hex_color)
        bg = Image.new("RGBA", output.size, rgb + (255,))
        bg.paste(output, (0, 0), output)
        del output; gc.collect()
        return bg.convert("RGB")
    elif output_type == "base64_bg":
        if bg_img is None: raise ValueError("Thiếu ảnh nền.")
        bg_img = bg_img.convert("RGBA").resize(output.size, Image.LANCZOS)
        bg_img.paste(output, (0, 0), output)
        del output; gc.collect()
        return bg_img
    else:
        raise ValueError("Chế độ không hợp lệ.")

def _cinematic_hdr(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    rgba = np.array(img); del img; gc.collect()
    rgb = rgba[..., :3]; alpha = rgba[..., 3] if rgba.shape[2] == 4 else None

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab); del lab
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    l_float = l.astype(np.float32) / 255.0
    l_corrected = np.power(l_float, 1.2) * 255.0
    l = np.clip(l_corrected, 0, 255).astype(np.uint8)
    lab = cv2.merge([l, a, b]); del l, a, b
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB); del lab
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv); del hsv
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v]); del h, s, v
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB); del hsv
    if alpha is not None:
        out = np.dstack((rgb, alpha)); del rgb, alpha
    else:
        out = rgb; del rgb
    result = Image.fromarray(out, "RGBA" if alpha is not None else "RGB")
    del out; gc.collect()
    return result

def _denoise_skin(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    rgba = np.array(img); del img; gc.collect()
    rgb = rgba[..., :3]; alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    filtered = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    if alpha is not None:
        out = np.dstack((filtered, alpha)); del filtered, alpha
    else:
        out = filtered; del filtered
    result = Image.fromarray(out, "RGBA" if alpha is not None else "RGB")
    del out; gc.collect()
    return result

# ---------- Pipeline ----------
def _pipeline_sync(pil_image: Image.Image, steps: List[dict]) -> Image.Image:
    img = pil_image
    for step in steps:
        action = step.get("action")
        params = step.get("params", {})
        if action == "remove_bg":
            output_type = params.get("output_type", "transparent")
            hex_color = params.get("hex_color")
            bg_img = None
            if output_type == "base64_bg" and "bg_image" in params:
                bg_b64 = params["bg_image"]
                if bg_b64.startswith("data:"):
                    bg_b64 = bg_b64.split(",", 1)[1]
                bg_bytes = base64.b64decode(bg_b64)
                bg_img = Image.open(io.BytesIO(bg_bytes))
            img = _remove_bg(img, output_type, hex_color, bg_img)
            if bg_img: del bg_img
        elif action == "super_res":
            scale = float(params.get("scale", 2.0))
            img = _super_resolution_chunked(img, scale)
        elif action == "hdr":
            img = _cinematic_hdr(img)
        elif action == "denoise":
            img = _denoise_skin(img)
        else:
            raise ValueError(f"Hành động không hợp lệ: {action}")
        gc.collect()
    return img

# ---------- Endpoint ----------
@app.post("/process")
async def process(
    image: UploadFile = File(...),
    steps: str = Form(...),
    output_format: str = Form("webp")
):
    img_bytes, _ = await read_and_validate_file(image)
    try:
        steps_list = json.loads(steps)
    except json.JSONDecodeError:
        raise HTTPException(400, "Danh sách bước xử lý không đúng định dạng JSON.")

    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGBA")
    except Exception:
        raise HTTPException(400, "Không thể đọc file ảnh.")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_pipeline_sync, pil_img, steps_list),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(500, "Quá thời gian xử lý (60s).")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Pipeline lỗi: {e}")
        raise HTTPException(500, "Xử lý ảnh thất bại.")
    finally:
        del pil_img; gc.collect()

    fmt = output_format.upper()
    if fmt not in ("PNG", "JPEG", "WEBP"):
        fmt = "WEBP"
    b64 = pil_to_base64(result, fmt)
    del result; gc.collect()
    return {"image_base64": f"data:image/{fmt.lower()};base64,{b64}"}

# ---------- Giao diện ----------
HTML_UI = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image SaaS Pro Max Ultra</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); min-height: 100vh; margin: 0; font-family: 'Inter', sans-serif; color: #f1f5f9; }
        .glass { background: rgba(255,255,255,0.08); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; }
        .drag-over { border-color: #3b82f6; background: rgba(59,130,246,0.15); }
        .slider-container { position: relative; width: 100%; height: 400px; overflow: hidden; border-radius: 16px; }
        .slider-container img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
        .img-after { clip-path: inset(0 calc(100% - var(--pos, 50%)) 0 0); }
        input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; background: #cbd5e1; border-radius: 5px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; background: #fff; border-radius: 50%; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }
        .terminal { background: #0f172a; color: #10b981; font-family: 'Courier New', monospace; padding: 12px; border-radius: 8px; max-height: 200px; overflow-y: auto; }
        .terminal .line { opacity: 0; animation: fadeIn 0.3s forwards; }
        @keyframes fadeIn { to { opacity: 1; } }
    </style>
</head>
<body class="flex items-center justify-center p-4">
<div class="glass w-full max-w-6xl p-6 md:p-8">
    <h1 class="text-3xl font-bold text-center mb-4">🚀 AI Image SaaS Pro Max Ultra</h1>
    <p class="text-center opacity-70 mb-6">Tách nền, Phóng to siêu nét, Cinematic HDR, Làm mịn da – Tất cả trong một Pipeline</p>
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div class="col-span-1 space-y-4">
            <div class="glass p-4 space-y-3">
                <h2 class="text-lg font-semibold">🧠 Chức năng</h2>
                <div class="flex flex-col gap-2">
                    <button onclick="addStep('remove_bg')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🎭 Tách nền</button>
                    <button onclick="addStep('super_res')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🔍 Phóng to siêu nét</button>
                    <button onclick="addStep('hdr')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🎬 Cinematic HDR</button>
                    <button onclick="addStep('denoise')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🧼 Làm mịn da</button>
                </div>
            </div>
            <div class="glass p-4 space-y-3">
                <h2 class="text-lg font-semibold">⚙️ Tham số</h2>
                <div id="param-container" class="space-y-2"></div>
            </div>
            <div class="glass p-4 space-y-3">
                <h2 class="text-lg font-semibold">📦 Định dạng xuất</h2>
                <select id="output-format" class="w-full bg-white/10 border border-white/20 rounded p-2 text-white">
                    <option value="webp" selected>WebP (siêu nhẹ, có trong suốt)</option>
                    <option value="png">PNG (không nén, chất lượng cao)</option>
                    <option value="jpeg">JPEG (ảnh nén, không trong suốt)</option>
                </select>
            </div>
            <button onclick="clearPipeline()" class="w-full py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg transition text-sm">🗑️ Xóa chuỗi xử lý</button>
        </div>
        <div class="col-span-3 space-y-6">
            <div id="drop-zone" class="border-2 border-dashed border-white/30 rounded-2xl p-8 text-center cursor-pointer transition hover:border-white/60">
                <i class="fas fa-cloud-upload-alt text-3xl mb-3"></i>
                <p class="text-lg font-medium">Kéo & thả ảnh vào đây</p>
                <p class="text-sm opacity-70">hoặc click để chọn (JPEG, PNG, WebP, tối đa 10MB)</p>
                <input type="file" id="file-input" accept="image/jpeg,image/png,image/webp" class="hidden">
            </div>
            <div id="pipeline-display" class="flex flex-wrap gap-2"></div>
            <button id="process-btn" class="w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-xl font-semibold transition flex items-center justify-center gap-2">
                <i class="fas fa-cogs"></i> Xử lý ngay
            </button>
            <div id="terminal" class="terminal text-sm" style="display:none;"></div>
            <div id="error-msg" class="hidden bg-red-500/80 p-3 rounded-xl"></div>
            <div id="result-section" class="hidden">
                <h3 class="text-lg font-semibold mb-3">So sánh kết quả</h3>
                <div class="slider-container">
                    <img id="img-before" src="" alt="Ảnh gốc">
                    <img id="img-after" src="" alt="Đã xử lý" class="img-after" style="--pos:50%">
                    <input type="range" id="compare-slider" min="0" max="100" value="50" class="absolute bottom-3 left-2 right-2">
                </div>
                <div class="flex justify-between mt-2 text-xs opacity-70"><span>Ảnh gốc</span><span>Đã xử lý</span></div>
                <div class="mt-3 text-center flex gap-3 justify-center">
                    <a id="download-link" href="#" download="processed_image" class="text-blue-400 hover:underline text-sm">📥 Tải ảnh đã xử lý</a>
                    <a id="download-original" href="#" download="original_image" class="text-blue-400 hover:underline text-sm">📥 Tải ảnh gốc</a>
                </div>
            </div>
        </div>
    </div>
</div>
<script>
    let pipeline = [];
    let currentFile = null, currentImageBase64 = null;
    const dropZone = document.getElementById('drop-zone'), fileInput = document.getElementById('file-input');
    dropZone.addEventListener('click', ()=> fileInput.click());
    dropZone.addEventListener('dragover', e=>{e.preventDefault(); dropZone.classList.add('drag-over');});
    dropZone.addEventListener('dragleave', ()=> dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e=>{
        e.preventDefault(); dropZone.classList.remove('drag-over');
        if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e=>{ if(e.target.files.length) handleFile(e.target.files[0]); });
    function handleFile(file){
        if(!file.type.startsWith('image/')){ showError('Vui lòng chọn file ảnh hợp lệ.'); return; }
        if(file.size > 10*1024*1024){ showError('Ảnh vượt quá 10MB.'); return; }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            currentImageBase64 = e.target.result;
            document.getElementById('img-before').src = e.target.result;
            document.getElementById('download-original').href = e.target.result;
            document.getElementById('download-original').download = file.name;
            document.getElementById('result-section').classList.add('hidden');
        };
        reader.readAsDataURL(file);
        hideError();
    }
    function showError(msg){ document.getElementById('error-msg').textContent = msg; document.getElementById('error-msg').classList.remove('hidden'); }
    function hideError(){ document.getElementById('error-msg').classList.add('hidden'); }
    function addStep(action){ pipeline.push({action, params:{}}); renderPipeline(); renderParams(); }
    function clearPipeline(){ pipeline=[]; renderPipeline(); renderParams(); }
    function removeStep(i){ pipeline.splice(i,1); renderPipeline(); renderParams(); }
    function renderPipeline(){
        document.getElementById('pipeline-display').innerHTML = pipeline.map((s,i)=>`
            <span class="px-3 py-1 bg-white/10 rounded-full text-sm flex items-center gap-1">
                ${({remove_bg:'🎭 Tách nền',super_res:'🔍 Phóng to',hdr:'🎬 HDR',denoise:'🧼 Làm mịn'}[s.action] || s.action)}
                <button onclick="removeStep(${i})" class="text-red-400 hover:text-red-600">&times;</button>
            </span>`).join('');
    }
    function renderParams(){
        const container = document.getElementById('param-container'); container.innerHTML = '';
        pipeline.forEach((step,idx)=>{
            if(step.action==='super_res'){
                const scale = step.params.scale || 2.0;
                container.innerHTML += <div class="flex items-center gap-2 text-sm"><span>🔍 Mức phóng (${idx+1}):</span><select onchange="updateParam(${idx},'scale',this.value)" class="bg-white/10 border border-white/20 rounded p-1"><option value="1.0" ${scale==1?'selected':''}>1x</option><option value="2.0" ${scale==2?'selected':''}>2x</option><option value="4.0" ${scale==4?'selected':''}>4x</option><option value="8.0" ${scale==8?'selected':''}>8x</option></select></div>;
            } else if(step.action==='remove_bg'){
                const outputType = step.params.output_type || 'transparent';
                container.innerHTML += <div class="flex items-center gap-2 text-sm"><span>🎨 Chế độ nền:</span><select onchange="updateParam(${idx},'output_type',this.value)" class="bg-white/10 border border-white/20 rounded p-1"><option value="transparent" ${outputType=='transparent'?'selected':''}>Trong suốt</option><option value="hex_color" ${outputType=='hex_color'?'selected':''}>Đổ màu</option><option value="base64_bg" ${outputType=='base64_bg'?'selected':''}>Ghép nền</option></select></div>;
                if(outputType==='hex_color'){
                    const hex = step.params.hex_color || '#FFFFFF';
                    container.innerHTML += <div class="flex items-center gap-2 text-sm"><span>Màu:</span><input type="text" value="${hex}" onchange="updateParam(${idx},'hex_color',this.value)" class="w-20 bg-white/10 border border-white/20 rounded p-1 text-white"></div>;
                }
                if(outputType==='base64_bg'){
                    container.innerHTML += <div class="text-sm"><label class="block">Ảnh nền:</label><input type="file" accept="image/*" onchange="handleBgUpload(${idx}, this)" class="mt-1 text-xs"><span id="bg-name-${idx}" class="text-xs opacity-70"></span></div>;
                }
            }
        });
    }
    function updateParam(idx, key, value){ pipeline[idx].params[key]=value; renderParams(); }
    function handleBgUpload(idx, input){
        if(input.files[0]){
            const reader = new FileReader();
            reader.onload = e => { pipeline[idx].params.bg_image = e.target.result; document.getElementById('bg-name-'+idx).textContent = input.files[0].name; };
            reader.readAsDataURL(input.files[0]);
        }
    }
    async function simulateTerminal(stepsList){
        const terminal = document.getElementById('terminal'); terminal.style.display='block';
        terminal.innerHTML = '<div class="text-green-400 font-bold">🔹 Bắt đầu xử lý...</div>';
        const msgs = { remove_bg: ['Đang phân tích pixel ảnh...','Đang tách nền bằng AI...','Hoàn tất tách nền.'], super_res: ['Đang áp dụng Lanczos-4 resize...','Đang tăng cường đường nét...','Phóng to thành công.'], hdr: ['Đang cân bằng sáng (CLAHE)...','Đang điều chỉnh Gamma...','Đang tăng cường màu sắc...','Hoàn tất Cinematic HDR.'], denoise: ['Đang khử nhiễu...','Giữ lại cạnh sắc nét...','Làm mịn da hoàn tất.'] };
        for(const step of stepsList){
            for(const line of (msgs[step.action]||['Đang xử lý...'])){
                await new Promise(r=>setTimeout(r,300));
                terminal.innerHTML += <div class="line">▹ ${line}</div>;
                terminal.scrollTop = terminal.scrollHeight;
            }
        }
        await new Promise(r=>setTimeout(r,200));
        terminal.innerHTML += '<div class="text-green-300 font-bold mt-1">✅ Xử lý hoàn tất!</div>';
    }
    document.getElementById('process-btn').addEventListener('click', async ()=>{
        if(!currentFile){ showError('Vui lòng tải ảnh lên trước.'); return; }
        if(pipeline.length===0){ showError('Thêm ít nhất một bước xử lý.'); return; }
        hideError(); document.getElementById('result-section').classList.add('hidden');
        await simulateTerminal(pipeline);
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('steps', JSON.stringify(pipeline));
        formData.append('output_format', document.getElementById('output-format').value);
        try {
            const res = await fetch('/process', {method:'POST',body:formData});
            const data = await res.json();
            if(!res.ok) throw new Error(data.message||'Lỗi');
            document.getElementById('img-after').src = data.image_base64;
            document.getElementById('img-before').src = currentImageBase64;
            document.getElementById('result-section').classList.remove('hidden');
            document.getElementById('compare-slider').value = 50;
            document.getElementById('img-after').style.setProperty('--pos','50%');
            document.getElementById('download-link').href = data.image_base64;
            document.getElementById('download-link').download = processed.${document.getElementById('output-format').value};
        } catch(err){ showError(err.message); }
    });
    document.getElementById('compare-slider').addEventListener('input', function(){
        document.getElementById('img-after').style.setProperty('--pos', this.value+'%');
    });
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_UI)

# ---------- Entry point ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
