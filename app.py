from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont
import io

try:
    import ultralytics
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    import ultralytics

app = FastAPI()

# Load YOLOv11 model (use a small model for demo, e.g., 'yolo11n.pt')
model = ultralytics.YOLO("yolo11n.pt")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Detector</title>
        <script src=\"https://cdn.tailwindcss.com\"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
    </head>
    <body class=\"bg-gray-100 min-h-screen flex flex-col items-center justify-center\">
        <div class=\"bg-white p-12 rounded shadow-md w-full max-w-6xl\">
            <h1 class=\"text-2xl font-bold mb-4 text-center\">Detector</h1>
            <div id="drop-zone" class="border-2 border-dashed border-gray-400 rounded-lg bg-gray-50 flex flex-col items-center justify-center p-8 cursor-pointer transition hover:bg-gray-100">
                <input id="file-input" type="file" name="file" accept="image/*" class="hidden" />
                <div id="drop-zone-content" class="flex flex-col items-center">
                    <i class="fa-solid fa-image text-5xl text-gray-400 mb-2"></i>
                    <span class="text-gray-500">Drag and drop an image file here or click</span>
                </div>
                <img id="preview" src="" alt="Preview" class="hidden mt-4 max-h-64 rounded shadow" />
            </div>
            <button id="detect-btn" class="bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition w-full mt-6 disabled:opacity-50" disabled>Detect Objects</button>
            <div id="result" class="mt-6 text-center"></div>
        </div>
        <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const detectBtn = document.getElementById('detect-btn');
        const resultDiv = document.getElementById('result');
        let selectedFile = null;

        // Drop zone click opens file dialog
        dropZone.addEventListener('click', () => fileInput.click());

        // Drag over styling
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('bg-blue-50');
        });
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('bg-blue-50');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('bg-blue-50');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) return;
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.classList.remove('hidden');
                document.getElementById('drop-zone-content').classList.add('hidden');
                detectBtn.disabled = false;
            };
            reader.readAsDataURL(file);
            // Clear previous result
            resultDiv.innerHTML = '';
        }

        detectBtn.onclick = async () => {
            if (!selectedFile) return;
            resultDiv.innerHTML = 'Processing...';
            const formData = new FormData();
            formData.append('file', selectedFile);
            const res = await fetch('/detect', {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                preview.classList.add('hidden'); // Hide preview after detection
                dropZone.classList.add('hidden'); // Hide drop zone after detection
                detectBtn.classList.add('hidden'); // Hide button after detection
                resultDiv.innerHTML = `<img src="${url}" class="mx-auto rounded shadow w-full h-[600px] object-contain" />` +
                  `<button id='try-another' class='mt-6 bg-blue-600 text-white py-2 px-6 rounded hover:bg-blue-700 transition'>Try another image</button>`;
                document.getElementById('try-another').onclick = () => location.reload();
            } else {
                resultDiv.innerHTML = 'Error processing image.';
            }
        };
        </script>
    </body>
    </html>
    """

@app.post("/detect")
def detect(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    # Calculate font size and box width based on image size
    font_size = max(int(height * 0.04), 12)
    box_width = max(int(width * 0.006), 2)
    results = model(image)
    boxes = results[0].boxes
    draw = ImageDraw.Draw(image)
    # Try to load a scalable font
    font_path = None
    possible_fonts = [
        "/Library/Fonts/Arial.ttf",  # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "arial.ttf",  # Windows or local
    ]
    for path in possible_fonts:
        try:
            font = ImageFont.truetype(path, font_size)
            font_path = path
            break
        except Exception:
            continue
    if not font_path:
        font = ImageFont.load_default()

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        draw.rectangle(xyxy, outline="red", width=box_width)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x, text_y = xyxy[0], max(xyxy[1] - text_height, 0)
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill="red"
        )
        # Draw black outline for better contrast
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), label, fill="black", font=font)
        draw.text((text_x, text_y), label, fill="white", font=font)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
