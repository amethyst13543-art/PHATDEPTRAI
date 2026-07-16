# ================================================
# app.py - AI Xử Lý Ảnh SaaS Production (Render)
# ================================================
from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import logging
import os
import re
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageColor, ImageFilter

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("saas-image-engine")

# ---------- Rate Limiter (chống spam/DDoS) ----------
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

# ---------- Model Singleton (tải một lần) ----------
_rmbg_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rmbg_session
    try:
        from rembg import new_session
        os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"
        _rmbg_session = new_session("u2netp")  # model nhẹ nhất
        logger.info("✅ Đã tải model tách nền (u2netp).")
    except Exception as e:
        logger.critical(f"❌ Không thể tải model: {e}")
        raise RuntimeError("Khởi tạo model thất bại.") from e
    yield
    if _rmbg_session:
        del _rmbg_session
    gc.collect()

# ---------- FastAPI App ----------
app = FastAPI(title="AI Image SaaS Pro", version="7.0.0", lifespan=lifespan)
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

async def read_and_validate_file(file: UploadFile, max_mb: int = 10) -> bytes:
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(415, "Chỉ chấp nhận ảnh JPEG, PNG hoặc WebP.")
    content = await file.read()
    if len(content) / (1024*1024) > max_mb:
        raise HTTPException(413, f"Ảnh vượt quá {max_mb}MB.")
    return content

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def limit_pixels(w: int, h: int, max_mp: int = 16):
    if w * h > max_mp * 1_000_000:
        raise HTTPException(413, f"Ảnh sau xử lý vượt quá {max_mp}MP, vui lòng giảm kích thước hoặc mức phóng.")

# ---------- Xử lý ảnh (đồng bộ, chạy trong thread) ----------
def _remove_bg(img_bytes: bytes, output_type: str = "transparent", hex_color: Optional[str] = None, bg_bytes: Optional[bytes] = None) -> Image.Image:
    from rembg import remove
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = img.size
    if w * h > 12_000_000:
        raise ValueError("Ảnh quá lớn (>12MP).")
    output = remove(img, session=_rmbg_session)
    del img; gc.collect()
    if output_type == "transparent":
        return output
    elif output_type == "hex_color":
        if not hex_color: raise ValueError("Thiếu mã màu.")
        validate_hex_color(hex_color)
        rgb = ImageColor.getrgb(hex_color)
        bg = Image.new("RGBA", output.size, rgb + (255,))
        bg.paste(output, (0,0), output)
        del output; gc.collect()
        return bg.convert("RGB")
    elif output_type == "base64_bg":
        if bg_bytes is None: raise ValueError("Thiếu ảnh nền.")
        bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGBA").resize(output.size, Image.LANCZOS)
        bg_img.paste(output, (0,0), output)
        del output; gc.collect()
        return bg_img
    else:
        raise ValueError("Chế độ không hợp lệ.")

def _super_resolution(img_bytes: bytes, scale: float = 2.0) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    limit_pixels(new_w, new_h, max_mp=16)  # giới hạn sau phóng
    # Lanczos4 resize
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Unsharp Masking
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    gc.collect()
    return img

def _cinematic_hdr(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    rgba = np.array(img); del img; gc.collect()
    rgb = rgba[..., :3]; alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    # CLAHE trên kênh L
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab); del lab
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    # Gamma correction (gamma=1.2)
    l_float = l.astype(np.float32) / 255.0
    l_corrected = np.power(l_float, 1.2) * 255.0
    l = np.clip(l_corrected, 0, 255).astype(np.uint8)
    lab = cv2.merge([l, a, b]); del l, a, b
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB); del lab
    # Tăng saturation nhẹ
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

def _denoise_skin(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    rgba = np.array(img); del img; gc.collect()
    rgb = rgba[..., :3]; alpha = rgba[..., 3] if rgba.shape[2] == 4 else None
    # Bilateral filter (giữ cạnh, làm mịn vùng phẳng)
    filtered = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    if alpha is not None:
        out = np.dstack((filtered, alpha)); del filtered, alpha
    else:
        out = filtered; del filtered
    result = Image.fromarray(out, "RGBA" if alpha is not None else "RGB")
    del out; gc.collect()
    return result

def _pipeline_sync(img_bytes: bytes, steps: List[dict]) -> Image.Image:
    """Xử lý ảnh qua chuỗi bước (mỗi bước có 'action' và 'params')."""
    current_bytes = img_bytes
    for step in steps:
        action = step.get("action")
        params = step.get("params", {})
        if action == "remove_bg":
            output_type = params.get("output_type", "transparent")
            hex_color = params.get("hex_color")
            bg_bytes = None
            if output_type == "base64_bg" and "bg_image" in params:
                bg_b64 = params["bg_image"]
                if bg_b64.startswith("data:"):
                    bg_b64 = bg_b64.split(",", 1)[1]
                bg_bytes = base64.b64decode(bg_b64)
            result = _remove_bg(current_bytes, output_type, hex_color, bg_bytes)
        elif action == "super_res":
            scale = float(params.get("scale", 2.0))
            result = _super_resolution(current_bytes, scale)
        elif action == "hdr":
            result = _cinematic_hdr(current_bytes)
        elif action == "denoise":
            result = _denoise_skin(current_bytes)
        else:
            raise ValueError(f"Hành động không hợp lệ: {action}")
        buf = io.BytesIO()
        fmt = params.get("format", "PNG")  # Cho từng bước xuất PNG để không mất chất lượng
        result.save(buf, format=fmt)
        current_bytes = buf.getvalue()
        del result, buf; gc.collect()
    return Image.open(io.BytesIO(current_bytes))

# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_UI)

@app.post("/process")
async def process(
    image: UploadFile = File(...),
    steps: str = Form(...),
    output_format: str = Form("webp")  # png, jpeg, webp
):
    img_bytes = await read_and_validate_file(image)
    try:
        steps_list = json.loads(steps)
    except json.JSONDecodeError:
        raise HTTPException(400, "Danh sách bước xử lý không đúng JSON.")
    try:
        result = await asyncio.to_thread(_pipeline_sync, img_bytes, steps_list)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline lỗi: {e}")
        raise HTTPException(500, "Xử lý thất bại.")
    # Xuất theo định dạng yêu cầu
    fmt = output_format.upper()
    if fmt not in ("PNG", "JPEG", "WEBP"):
        fmt = "WEBP"
    b64 = pil_to_base64(result, fmt)
    del result; gc.collect()
    return {"image_base64": f"data:image/{fmt.lower()};base64,{b64}"}

# ---------- Giao diện SaaS siêu thực ----------
HTML_UI = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image SaaS Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            margin: 0;
            font-family: 'Inter', sans-serif;
            color: #f1f5f9;
        }
        .glass {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
        }
        .drag-over { border-color: #3b82f6; background: rgba(59,130,246,0.15); }
        .slider-container { position: relative; width: 100%; height: 360px; overflow: hidden; border-radius: 16px; }
        .slider-container img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
        .img-after { clip-path: inset(0 calc(100% - var(--pos, 50%)) 0 0); }
        input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; background: #cbd5e1; border-radius: 5px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; background: #fff; border-radius: 50%; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }
        .btn-active { background: rgba(255,255,255,0.2); }
        .terminal { background: #0f172a; color: #10b981; font-family: 'Courier New', monospace; padding: 12px; border-radius: 8px; max-height: 200px; overflow-y: auto; }
        .terminal .line { opacity: 0; animation: fadeIn 0.3s forwards; }
        @keyframes fadeIn { to { opacity: 1; } }
    </style>
</head>
<body class="flex items-center justify-center p-4">
<div class="glass w-full max-w-6xl p-6 md:p-8">
    <h1 class="text-3xl font-bold text-center mb-6">🚀 AI Image SaaS Pro</h1>
    <!-- Bảng điều khiển -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <!-- Sidebar chức năng -->
        <div class="col-span-1 space-y-4">
            <div class="glass p-4 space-y-3">
                <h2 class="text-lg font-semibold">🧠 Chức năng</h2>
                <div class="flex flex-col gap-2">
                    <button onclick="addStep('remove_bg')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🎭 Tách nền</button>
                    <button onclick="addStep('super_res')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🔍 Phóng to siêu nét</button>
                    <button onclick="addStep('hdr')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🎬 Cinematic HDR</button>
                    <button onclick="addStep('denoise')" class="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition text-sm">🧼 Làm mịn da / Khử nhiễu</button>
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
        <!-- Khu vực chính -->
        <div class="col-span-3 space-y-6">
            <!-- Kéo thả -->
            <div id="drop-zone" class="border-2 border-dashed border-white/30 rounded-2xl p-8 text-center cursor-pointer transition hover:border-white/60">
                <i class="fas fa-cloud-upload-alt text-3xl mb-3"></i>
                <p class="text-lg font-medium">Kéo & thả ảnh vào đây</p>
                <p class="text-sm opacity-70">hoặc click để chọn (JPEG, PNG, WebP, tối đa 10MB)</p>
                <input type="file" id="file-input" accept="image/jpeg,image/png,image/webp" class="hidden">
            </div>
            <!-- Chuỗi xử lý -->
            <div id="pipeline-display" class="flex flex-wrap gap-2"></div>
            <!-- Nút xử lý & terminal -->
            <button id="process-btn" class="w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-xl font-semibold transition flex items-center justify-center gap-2">
                <i class="fas fa-cogs"></i> Xử lý ngay
            </button>
            <div id="terminal" class="terminal text-sm" style="display:none;"></div>
            <!-- Thông báo lỗi -->
            <div id="error-msg" class="hidden bg-red-500/80 p-3 rounded-xl"></div>
            <!-- Kết quả so sánh -->
            <div id="result-section" class="hidden">
                <h3 class="text-lg font-semibold mb-3">So sánh kết quả</h3>
                <div class="slider-container">
                    <img id="img-before" src="" alt="Ảnh gốc">
                    <img id="img-after" src="" alt="Đã xử lý" class="img-after" style="--pos:50%">
                    <input type="range" id="compare-slider" min="0" max="100" value="50" class="absolute bottom-3 left-2 right-2">
                </div>
                <div class="flex justify-between mt-2 text-xs opacity-70">
                    <span>Ảnh gốc</span><span>Đã xử lý</span>
                </div>
                <div class="mt-3 text-center">
                    <a id="download-link" href="#" download="processed_image" class="text-blue-400 hover:underline text-sm">📥 Tải ảnh về</a>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    // State
    let pipeline = [];
    let currentFile = null;
    let currentImageBase64 = null;

    // Kéo thả
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    dropZone.addEventListener('click', ()=> fileInput.click());
    dropZone.addEventListener('dragover', e=>{e.preventDefault(); dropZone.classList.add('drag-over');});
    dropZone.addEventListener('dragleave', ()=> dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e=>{
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e=>{
        if(e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file){
        if(!file.type.startsWith('image/')){
            showError('Vui lòng chọn file ảnh hợp lệ.');
            return;
        }
        if(file.size > 10*1024*1024){
            showError('Ảnh vượt quá 10MB.');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            currentImageBase64 = e.target.result;
            document.getElementById('img-before').src = e.target.result;
            document.getElementById('result-section').classList.add('hidden');
        };
        reader.readAsDataURL(file);
        hideError();
    }

    function showError(msg){
        const el = document.getElementById('error-msg');
        el.textContent = msg;
        el.classList.remove('hidden');
    }
    function hideError(){ document.getElementById('error-msg').classList.add('hidden'); }

    // Pipeline builder
    function addStep(action){
        const step = { action, params: {} };
        pipeline.push(step);
        renderPipeline();
        renderParams();
    }
    function clearPipeline(){
        pipeline = [];
        renderPipeline();
        renderParams();
    }
    function removeStep(index){
        pipeline.splice(index,1);
        renderPipeline();
        renderParams();
    }
    function renderPipeline(){
        const container = document.getElementById('pipeline-display');
        container.innerHTML = pipeline.map((s,i)=>`
            <span class="px-3 py-1 bg-white/10 rounded-full text-sm flex items-center gap-1">
                ${actionLabel(s.action)}
                <button onclick="removeStep(${i})" class="text-red-400 hover:text-red-600">&times;</button>
            </span>
        `).join('');
    }
    function actionLabel(action){
        const map = {
            'remove_bg':'🎭 Tách nền',
            'super_res':'🔍 Phóng to',
            'hdr':'🎬 HDR',
            'denoise':'🧼 Làm mịn'
        };
        return map[action] || action;
    }
    function renderParams(){
        const container = document.getElementById('param-container');
        container.innerHTML = '';
        pipeline.forEach((step, idx)=>{
            if(step.action === 'super_res'){
                const scale = step.params.scale || 2.0;
                container.innerHTML += `
                <div class="flex items-center gap-2 text-sm">
                    <span>🔍 Mức phóng (${idx+1}):</span>
                    <select onchange="updateParam(${idx},'scale',this.value)" class="bg-white/10 border border-white/20 rounded p-1">
                        <option value="1.0" ${scale==1?'selected':''}>1x</option>
                        <option value="2.0" ${scale==2?'selected':''}>2x</option>
                        <option value="4.0" ${scale==4?'selected':''}>4x</option>
                        <option value="8.0" ${scale==8?'selected':''}>8x</option>
                    </select>
                </div>`;
            } else if(step.action === 'remove_bg'){
                const outputType = step.params.output_type || 'transparent';
                container.innerHTML += `
                <div class="flex items-center gap-2 text-sm">
                    <span>🎨 Chế độ nền:</span>
                    <select onchange="updateParam(${idx},'output_type',this.value)" class="bg-white/10 border border-white/20 rounded p-1">
                        <option value="transparent" ${outputType=='transparent'?'selected':''}>Trong suốt</option>
                        <option value="hex_color" ${outputType=='hex_color'?'selected':''}>Đổ màu</option>
                        <option value="base64_bg" ${outputType=='base64_bg'?'selected':''}>Ghép nền</option>
                    </select>
                </div>`;
                if(outputType === 'hex_color'){
                    const hex = step.params.hex_color || '#FFFFFF';
                    container.innerHTML += `
                    <div class="flex items-center gap-2 text-sm">
                        <span>Màu:</span>
                        <input type="text" value="${hex}" onchange="updateParam(${idx},'hex_color',this.value)" class="w-20 bg-white/10 border border-white/20 rounded p-1 text-white">
                    </div>`;
                }
                if(outputType === 'base64_bg'){
                    container.innerHTML += `
                    <div class="text-sm">
                        <label class="block">Ảnh nền:</label>
                        <input type="file" accept="image/*" onchange="handleBgUpload(${idx}, this)" class="mt-1 text-xs">
                        <span id="bg-name-${idx}" class="text-xs opacity-70"></span>
                    </div>`;
                }
            }
        });
    }
    function updateParam(idx, key, value){
        pipeline[idx].params[key] = value;
        renderParams();
    }
    function handleBgUpload(idx, input){
        const file = input.files[0];
        if(file){
            const reader = new FileReader();
            reader.onload = e => {
                pipeline[idx].params.bg_image = e.target.result;
                document.getElementById('bg-name-'+idx).textContent = file.name;
            };
            reader.readAsDataURL(file);
        }
    }

    // Terminal mô phỏng
    async function simulateTerminal(stepsList){
        const terminal = document.getElementById('terminal');
        terminal.style.display = 'block';
        terminal.innerHTML = '<div class="text-green-400 font-bold">🔹 Bắt đầu xử lý...</div>';
        const msgs = {
            'remove_bg': ['Đang phân tích pixel ảnh...', 'Đang tách nền bằng AI...', 'Hoàn tất tách nền.'],
            'super_res': ['Đang áp dụng Lanczos-4 resize...', 'Đang tăng cường đường nét (Unsharp Masking)...', 'Phóng to thành công.'],
            'hdr': ['Đang cân bằng sáng (CLAHE)...', 'Đang điều chỉnh Gamma...', 'Đang tăng cường màu sắc...', 'Hoàn tất Cinematic HDR.'],
            'denoise': ['Đang khử nhiễu (Bilateral Filter)...', 'Giữ lại cạnh sắc nét...', 'Làm mịn da hoàn tất.']
        };
        for (const step of stepsList) {
            const lines = msgs[step.action] || ['Đang xử lý...'];
            for (const line of lines) {
                await delay(300);
                terminal.innerHTML += <div class="line">▹ ${line}</div>;
                terminal.scrollTop = terminal.scrollHeight;
            }
        }
        await delay(200);
        terminal.innerHTML += '<div class="text-green-300 font-bold mt-1">✅ Xử lý hoàn tất!</div>';
    }
    function delay(ms){ return new Promise(r=>setTimeout(r,ms)); }

    // Xử lý chính
    document.getElementById('process-btn').addEventListener('click', async ()=>{
        if(!currentFile){
            showError('Vui lòng tải ảnh lên trước.');
            return;
        }
        if(pipeline.length === 0){
            showError('Thêm ít nhất một bước xử lý.');
            return;
        }
        hideError();
        const resultSection = document.getElementById('result-section');
        resultSection.classList.add('hidden');
        // Terminal
        await simulateTerminal(pipeline);
        // Gửi request
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('steps', JSON.stringify(pipeline));
        formData.append('output_format', document.getElementById('output-format').value);
        try {
            const res = await fetch('/process', { method: 'POST', body: formData });
            const data = await res.json();
            if(!res.ok) throw new Error(data.message || 'Lỗi');
            const imgAfter = document.getElementById('img-after');
            imgAfter.src = data.image_base64;
            document.getElementById('img-before').src = currentImageBase64;
            resultSection.classList.remove('hidden');
            document.getElementById('compare-slider').value = 50;
            imgAfter.style.setProperty('--pos','50%');
            // Link tải về
            const link = document.getElementById('download-link');
            link.href = data.image_base64;
            const ext = document.getElementById('output-format').value;
            link.download = processed.${ext};
        } catch(err){
            showError(err.message);
        }
    });

    // Slider so sánh
    document.getElementById('compare-slider').addEventListener('input', function(){
        document.getElementById('img-after').style.setProperty('--pos', this.value + '%');
    });
</script>
</body>
</html>
"""

# ---------- Entry point ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
