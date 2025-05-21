# Detector
A simple Python web app for object detection using Ultralytics YOLOv11. Upload an image and get bounding boxes and confidence scores in a modern, responsive web interface styled with TailwindCSS.

## Features
- Upload images for object detection
- Bounding boxes and confidence scores drawn on the image
- FastAPI backend, HTML+TailwindCSS frontend
- No JavaScript build step required

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/fromtheroot/detector.git
   cd detector
   ```

2. **Create a conda environment (Python 3.11)**
   ```sh
   conda create --name yolo python=3.11
   conda activate yolo
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```sh
   python app.py
   ```

5. **Open your browser**
   Go to [http://localhost:8000](http://localhost:8000) to use the web UI.

## Notes
- The app will automatically attempt to download the YOLOv11 model weights (`yolo11n.pt`) if not present.
- If you want to use a different YOLO model, change the model path in `app.py`.
- The app uses system fonts for label rendering.