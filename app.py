from _future_ import annotations

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
from PIL import Image, ImageColor

# ================================================
# Logging
# ================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vision-ai")

# ================================================
# Rate Limiter (chống spam/DDoS)
# ================================================
class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def is_allowed(self, ip: str) -> bool:
        async with self.lock:
            now = time.time()
            # Xóa các timestamp cũ
            self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
            if len(self.requests[ip]) >= self.max_requests:
                return False
            self.requests[ip].append(now)
            return True

rate_limiter = RateLimiter(max_requests=10, window=60)

# ================================================
# Model Singleton (tải một lần khi startup)
# ================================================
_rmbg_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rmbg_session
    try:
        from rembg import new_session
        os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"
        _rmbg_session = new_session("u2netp")   # nhẹ nhất (≈4MB)
        logger.info("✅ Đã tải model tách nền (u2netp).")
    except Exception as e:
        logger.critical(f"❌ Không thể tải model: {e}")
        raise RuntimeError("Khởi tạo model thất bại. Ứng dụng không thể hoạt động.") from e
    yield
    # Cleanup
    if _rmbg_session:
        del _rmbg_session
    gc.collect()

# ================================================
# FastAPI App
# ================================================
app = FastAPI(
    title="AI Xử Lý Ảnh Thế Hệ Mới",
    description="Tách nền, làm nét, cân bằng sáng tự động, pipeline liên hoàn",
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================
# Exception handler toàn cục
# ================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": True, "message": exc.detail})
    logger.error(f"Ngoại lệ không xử lý: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error": True, "message": "Lỗi máy chủ nội bộ. Vui lòng thử lại sau."})

# ================================================
# Rate Limiting Middleware (chỉ áp dụng POST)
# ================================================
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method == "POST":
        client_ip = request.client.host if request.client else "127.0.0.1"
        if not await rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": True, "message": "IP của bạn thao tác quá nhanh. Vui lòng thử lại sau 1 phút."},
            )
    response = await call_next(request)
    return response

# ================================================
# Helpers
# ================================================
def validate_hex_color(color: str) -> str:
    if not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$", color):
        raise ValueError(f"Mã màu không hợp lệ: {color}")
    return color.upper()

async def read_and_validate_file(file: UploadFile, max_mb: int = 10) -> bytes:
    """Đọc file upload, kiểm tra MIME type và kích thước."""
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Chỉ chấp nhận ảnh JPEG, PNG hoặc WebP.")
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(status_code=413, detail=f"Ảnh vượt quá {max_mb}MB ({size_mb:.1f}MB).")
    return content

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ================================================
# Các hàm xử lý ảnh (đồng bộ, chạy trong thread)
# ================================================
def _remove_background_sync(
    img_bytes: bytes,
    output_type: str = "transparent",
    hex_color: Optional[str] = None,
    bg_bytes: Optional[bytes] = None,
) -> Image.Image:
    """Tách nền ảnh. Trả về PIL Image."""
    from rembg import remove
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = img.size
    if w * h > 12_000_000:  # 12 MP
        raise ValueError("Ảnh có độ phân giải quá lớn (>12MP), vui lòng giảm kích thước.")

    output = remove(img, session=_rmbg_session)  # PIL Image RGBA

    if output_type == "transparent":
        return output
    elif output_type == "hex_color":
        validate_hex_color(hex_color)
        rgb = ImageColor.getrgb(hex_color)
        background = Image.new("RGBA", output.size, rgb + (255,))
        background.paste(output, (0, 0), output)
        return background.convert("RGB")
    elif output_type == "base64_bg":
        if bg_bytes is None:
            raise ValueError("Thiếu ảnh nền để ghép.")
        bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
        bg_img = bg_img.resize(output.size, Image.LANCZOS)
        bg_img.paste(output, (0, 0), output)
        return bg_img
    else:
        raise ValueError(f"Chế độ nền không hợp lệ: {output_type}")

def _auto_retouch_sync(img_bytes: bytes) -> Image.Image:
    """
    Tự động cân bằng sáng/màu (Auto HDR effect).
    Sử dụng OpenCV CLAHE + tăng saturation.
    """
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    rgba = np.array(pil_img)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else None

    # Tăng saturation trong không gian HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.3, 0, 255)  # tăng 30%
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # CLAHE trên kênh L của LAB để cải thiện tương phản
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Ghép lại kênh alpha nếu có
    if alpha is not None:
        out = np.dstack((rgb, alpha))
    else:
        out = rgb
    return Image.fromarray(out, "RGBA" if alpha is not None else "RGB")

def _enhance_sync(img_bytes: bytes, upscale_factor: float = 2.0, face_enhance: bool = False) -> Image.Image:
    """Làm nét ảnh (placeholder)."""
    raise NotImplementedError("Tính năng làm nét ảnh đang được phát triển, mong bạn quay lại sau.")

def _pipeline_sync(img_bytes: bytes, steps: List[dict]) -> Image.Image:
    """Xử lý ảnh qua chuỗi các bước (mỗi bước là một dict có 'action' và 'params')."""
    current_bytes = img_bytes
    for step in steps:
        action = step.get("action")
        params = step.get("params", {})

        if action == "remove_bg":
            output_type = params.get("output_type", "transparent")
            hex_color = params.get("hex_color", None)
            bg_bytes = None
            if output_type == "base64_bg" and "bg_image" in params:
                # bg_image là base64 string từ giao diện (nếu có)
                bg_b64 = params["bg_image"]
                if bg_b64.startswith("data:"):
                    bg_b64 = bg_b64.split(",", 1)[1]
                bg_bytes = base64.b64decode(bg_b64)
            result = _remove_background_sync(current_bytes, output_type, hex_color, bg_bytes)
        elif action == "retouch":
            result = _auto_retouch_sync(current_bytes)
        elif action == "enhance":
            # Hiện tại chưa hỗ trợ
            raise NotImplementedError("Tính năng làm nét chưa sẵn sàng trong pipeline.")
        else:
            raise ValueError(f"Hành động không hợp lệ: {action}")

        # Lưu kết quả thành bytes để bước sau sử dụng
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        current_bytes = buf.getvalue()
        gc.collect()  # dọn rác sau mỗi bước

    # Trả về ảnh cuối cùng
    return Image.open(io.BytesIO(current_bytes))

# ================================================
# Giao diện Web siêu thực (Glassmorphism)
# ================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_UI)

HTML_UI = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Xử Lý Ảnh Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', sans-serif;
        }
        .glass {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .drag-over {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }
        .slider-container {
            position: relative;
            width: 100%;
            height: 300px;
            overflow: hidden;
        }
        .slider-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .img-after {
            clip-path: inset(0 calc(100% - var(--pos, 50%)) 0 0);
        }
        input[type=range] {
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            height: 6px;
            background: #fff;
            border-radius: 5px;
            outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 22px;
            height: 22px;
            background: #fff;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        .btn-active {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
<div class="glass w-full max-w-4xl p-6 md:p-8 text-white">
    <h1 class="text-3xl font-bold text-center mb-6">✨ AI Xử Lý Ảnh Thế Hệ Mới</h1>

    <!-- Chế độ xử lý -->
    <div class="flex flex-wrap gap-2 mb-6 justify-center">
        <button id="btn-remove" onclick="setMode('remove_bg')" class="px-4 py-2 rounded-full bg-white/20 hover:bg-white/30 transition">🎭 Tách nền</button>
        <button id="btn-retouch" onclick="setMode('retouch')" class="px-4 py-2 rounded-full bg-white/20 hover:bg-white/30">🌟 Cân bằng sáng</button>
        <button id="btn-pipeline" onclick="setMode('pipeline')" class="px-4 py-2 rounded-full bg-white/20 hover:bg-white/30">🔗 Combo</button>
    </div>

    <!-- Khu vực kéo thả ảnh -->
    <div id="drop-zone" class="border-2 border-dashed border-white/50 rounded-2xl p-8 text-center cursor-pointer mb-6 transition hover:border-white/80">
        <i class="fas fa-cloud-upload-alt text-4xl mb-2"></i>
        <p class="text-lg font-medium">Kéo & thả ảnh vào đây</p>
        <p class="text-sm opacity-75">hoặc click để chọn file (JPEG, PNG, WebP, tối đa 10MB)</p>
        <input type="file" id="file-input" accept="image/jpeg,image/png,image/webp" class="hidden">
    </div>

    <!-- Thông báo lỗi -->
    <div id="error-msg" class="hidden bg-red-500/80 text-white p-3 rounded-xl mb-4"></div>

    <!-- Cấu hình cho Tách nền -->
    <div id="config-remove" class="hidden space-y-3 mb-4">
        <select id="output-type" class="w-full p-2 rounded bg-white/20 text-white border border-white/30">
            <option value="transparent">Nền trong suốt</option>
            <option value="hex_color">Đổ màu nền</option>
            <option value="base64_bg">Ghép với ảnh nền</option>
        </select>
        <div id="hex-group" class="hidden">
            <input type="text" id="hex-color" value="#FFFFFF" class="w-full p-2 rounded bg-white/20 text-white border border-white/30 placeholder-white/70" placeholder="Mã màu hex">
        </div>
        <div id="bg-group" class="hidden">
            <label class="block text-sm mb-1">Ảnh nền:</label>
            <input type="file" id="bg-file" accept="image/*" class="w-full p-2 rounded bg-white/20 text-white border border-white/30">
        </div>
    </div>

    <!-- Cấu hình Pipeline -->
    <div id="config-pipeline" class="hidden space-y-3 mb-4">
        <div class="flex flex-wrap gap-2" id="pipeline-steps"></div>
        <div class="flex gap-2">
            <button onclick="addStep('retouch')" class="px-3 py-1 bg-white/20 rounded-full text-sm">+ Cân bằng sáng</button>
            <button onclick="addStep('remove_bg')" class="px-3 py-1 bg-white/20 rounded-full text-sm">+ Tách nền</button>
            <button onclick="clearSteps()" class="px-3 py-1 bg-red-400/50 rounded-full text-sm">Xóa tất cả</button>
        </div>
    </div>

    <!-- Nút xử lý & thanh tiến trình -->
    <button id="process-btn" class="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 rounded-full transition mb-4">🚀 Xử lý ngay</button>
    <div id="progress" class="hidden w-full bg-gray-200/30 rounded-full h-3 mb-4">
        <div class="bg-gradient-to-r from-blue-400 to-purple-500 h-3 rounded-full animate-pulse w-full"></div>
    </div>

    <!-- Kết quả & so sánh Before/After -->
    <div id="result-section" class="hidden">
        <h3 class="text-lg font-semibold mb-2">Kết quả</h3>
        <div class="slider-container" id="compare-container">
            <img id="img-before" src="" alt="Ảnh gốc">
            <img id="img-after" src="" alt="Đã xử lý" class="img-after" style="--pos:50%">
            <input type="range" id="compare-slider" min="0" max="100" value="50" class="absolute bottom-2 left-0 right-0 mx-4">
        </div>
        <div class="flex justify-between mt-2">
            <span class="text-xs">Ảnh gốc</span>
            <span class="text-xs">Đã xử lý</span>
        </div>
    </div>
</div>

<script>
    let currentMode = 'remove_bg';
    let currentFile = null;
    let pipelineSteps = [];

    function setMode(mode) {
        currentMode = mode;
        document.getElementById('config-remove').classList.toggle('hidden', mode !== 'remove_bg');
        document.getElementById('config-pipeline').classList.toggle('hidden', mode !== 'pipeline');
        ['btn-remove','btn-retouch','btn-pipeline'].forEach(id => {
            document.getElementById(id).classList.remove('btn-active');
        });
        document.getElementById('btn-' + (mode==='remove_bg'?'remove':mode==='retouch'?'retouch':'pipeline')).classList.add('btn-active');
    }

    // Kéo thả
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Vui lòng chọn file ảnh hợp lệ.');
            return;
        }
        if (file.size > 10*1024*1024) {
            showError('Ảnh vượt quá 10MB, vui lòng chọn ảnh nhỏ hơn.');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            document.getElementById('img-before').src = e.target.result;
            document.getElementById('result-section').classList.add('hidden');
        };
        reader.readAsDataURL(file);
        document.getElementById('error-msg').classList.add('hidden');
    }

    function showError(msg) {
        const el = document.getElementById('error-msg');
        el.textContent = msg;
        el.classList.remove('hidden');
    }

    // Hiển thị/ẩn trường tùy chọn tách nền
    document.getElementById('output-type').addEventListener('change', function() {
        document.getElementById('hex-group').classList.add('hidden');
        document.getElementById('bg-group').classList.add('hidden');
        if (this.value === 'hex_color') document.getElementById('hex-group').classList.remove('hidden');
        if (this.value === 'base64_bg') document.getElementById('bg-group').classList.remove('hidden');
    });

    // Pipeline builder
    function addStep(action) {
        pipelineSteps.push({action, params: {}});
        renderPipeline();
    }
    function clearSteps() {
        pipelineSteps = [];
        renderPipeline();
    }
    function renderPipeline() {
        const container = document.getElementById('pipeline-steps');
        container.innerHTML = pipelineSteps.map((s,i) => 
            `<span class="px-3 py-1 bg-white/20 rounded-full text-sm flex items-center gap-1">
                ${s.action==='retouch'?'🌟 Cân bằng sáng':'🎭 Tách nền'}
                <button onclick="removeStep(${i})" class="text-red-300 hover:text-red-500">&times;</button>
            </span>`
        ).join('');
    }
    function removeStep(index) {
        pipelineSteps.splice(index,1);
        renderPipeline();
    }

    // Xử lý chính
    document.getElementById('process-btn').addEventListener('click', async () => {
        if (!currentFile) {
            showError('Vui lòng tải ảnh lên trước.');
            return;
        }
        const progress = document.getElementById('progress');
        const resultSection = document.getElementById('result-section');
        progress.classList.remove('hidden');
        resultSection.classList.add('hidden');
        document.getElementById('error-msg').classList.add('hidden');

        const formData = new FormData();
        formData.append('image', currentFile);
        let url = '/process';

        if (currentMode === 'remove_bg') {
            url = '/remove-bg';
            const outputType = document.getElementById('output-type').value;
            formData.append('output_type', outputType);
            if (outputType === 'hex_color')
                formData.append('hex_color', document.getElementById('hex-color').value);
            if (outputType === 'base64_bg') {
                const bgFile = document.getElementById('bg-file').files[0];
                if (!bgFile) {
                    showError('Vui lòng chọn ảnh nền.');
                    progress.classList.add('hidden');
                    return;
                }
                formData.append('bg_image', bgFile);
            }
        } else if (currentMode === 'retouch') {
            url = '/retouch';
        } else if (currentMode === 'pipeline') {
            if (pipelineSteps.length === 0) {
                showError('Thêm ít nhất một bước xử lý cho combo.');
                progress.classList.add('hidden');
                return;
            }
            formData.append('steps', JSON.stringify(pipelineSteps));
        }

        try {
            const res = await fetch(url, { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.message || 'Lỗi không xác định');

            // Hiển thị ảnh kết quả và bật slider so sánh
            document.getElementById('img-after').src = data.image_base64;
            document.getElementById('img-before').src = URL.createObjectURL(currentFile);
            document.getElementById('result-section').classList.remove('hidden');
            document.getElementById('compare-slider').value = 50;
            document.getElementById('img-after').style.setProperty('--pos', '50%');
        } catch (err) {
            showError(err.message);
        } finally {
            progress.classList.add('hidden');
        }
    });

    // Slider so sánh
    document.getElementById('compare-slider').addEventListener('input', function() {
        document.getElementById('img-after').style.setProperty('--pos', this.value + '%');
    });
</script>
</body>
</html>
"""

# ================================================
# Endpoints xử lý ảnh
# ================================================
@app.post("/remove-bg")
async def remove_bg(
    image: UploadFile = File(...),
    output_type: str = Form("transparent"),
    hex_color: Optional[str] = Form(None),
    bg_image: Optional[UploadFile] = File(None),
):
    img_bytes = await read_and_validate_file(image)
    bg_bytes = None
    if bg_image:
        bg_bytes = await read_and_validate_file(bg_image)

    try:
        result = await asyncio.to_thread(
            _remove_background_sync, img_bytes, output_type, hex_color, bg_bytes
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi tách nền: {e}")
        raise HTTPException(status_code=500, detail="Xử lý tách nền thất bại.")

    b64 = pil_to_base64(result, "PNG")
    gc.collect()
    return {"image_base64": f"data:image/png;base64,{b64}"}

@app.post("/retouch")
async def retouch(image: UploadFile = File(...)):
    img_bytes = await read_and_validate_file(image)
    try:
        result = await asyncio.to_thread(_auto_retouch_sync, img_bytes)
    except Exception as e:
        logger.error(f"Lỗi cân bằng sáng: {e}")
        raise HTTPException(status_code=500, detail="Không thể cân bằng sáng ảnh.")

    b64 = pil_to_base64(result, "PNG")
    gc.collect()
    return {"image_base64": f"data:image/png;base64,{b64}"}

@app.post("/enhance")
async def enhance(image: UploadFile = File(...), upscale_factor: float = Form(2.0), face_enhance: bool = Form(False)):
    img_bytes = await read_and_validate_file(image)
    try:
        result = await asyncio.to_thread(_enhance_sync, img_bytes, upscale_factor, face_enhance)
        b64 = pil_to_base64(result, "PNG")
        gc.collect()
        return {"image_base64": f"data:image/png;base64,{b64}"}
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

@app.post("/process")
async def process_pipeline(image: UploadFile = File(...), steps: str = Form(...)):
    img_bytes = await read_and_validate_file(image)
    try:
        steps_list = json.loads(steps)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Danh sách bước xử lý không đúng định dạng JSON.")
    try:
        result = await asyncio.to_thread(_pipeline_sync, img_bytes, steps_list)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi pipeline: {e}")
        raise HTTPException(status_code=500, detail="Pipeline xử lý thất bại.")

    b64 = pil_to_base64(result, "PNG")
    gc.collect()
    return {"image_base64": f"data:image/png;base64,{b64}"}

# ================================================
# Entry point
# ================================================
if _name_ == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
