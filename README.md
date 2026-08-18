# Number Plate Detection Project

This project detects vehicles, reads license plates, and checks them against a local database.

## 1) Install Python

Use Python 3.10 or 3.11 on Windows.

Recommended:
- Python 3.11.x
- Git
- Visual Studio Build Tools (only if a native dependency fails to build)

Check the version:

```bash
python --version
```

If Python is not installed, download it from:
https://www.python.org/downloads/windows/

---

## 2) Clone the project

```bash
git clone https://github.com/PeddiAravind/opencv_numberplate_detection.git
cd opencv_numberplate_detection
```

---

## 3) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

Confirm the environment is active. Your terminal should show `(venv)`.

---

## 4) Install project dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails because of PyTorch or EasyOCR on your machine, install a compatible Python version first and retry.

---

## 5) Make sure required local files exist

This project expects these files to be present in the project folder:

- `models/license_plate_detector.pt`
- `videos/sample.mp4`
- `sounds/alert.wav`
- `database/vehicles.csv`

These files are usually large or local data files and may not be pushed to GitHub. If they are missing, place them manually before running the app.

Example structure:

```text
project-root/
├── models/
│   └── license_plate_detector.pt
├── videos/
│   └── sample.mp4
├── sounds/
│   └── alert.wav
├── database/
│   └── vehicles.csv
├── main.py
├── detector.py
├── requirements.txt
└── ...
```

### If the YOLO base model is missing

The project also expects a base model file named `yolov8n.pt` in the root folder.

You can download it automatically once with:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This should create the `yolov8n.pt` file in the project root.

---

## 6) Run the app

From the project root:

```bash
python main.py
```

The app will:
- open the video
- detect vehicles
- detect number plates
- read plate text using OCR
- compare plate numbers against the CSV database
- save detections and results

---

## 7) Common issues and fixes

### Problem: `ModuleNotFoundError`

Run:

```bash
pip install -r requirements.txt
```

### Problem: camera/video file not found

Make sure `videos/sample.mp4` exists.

### Problem: model file not found

Make sure these files exist:

```text
models/license_plate_detector.pt
yolov8n.pt
```

### Problem: audio alert file missing

Add or restore `sounds/alert.wav`.

### Problem: `torch` / `easyocr` install issues

Use Python 3.10 or 3.11, not a newer unsupported version.

---

## 8) Recommended final setup checklist

Before running the project, check:

```text
[ ] Python 3.10 or 3.11 installed
[ ] venv activated
[ ] pip install -r requirements.txt completed
[ ] yolov8n.pt exists
[ ] models/license_plate_detector.pt exists
[ ] videos/sample.mp4 exists
[ ] sounds/alert.wav exists
[ ] database/vehicles.csv exists
[ ] main.py runs without errors
```

---

## 9) Useful commands

```bash
python --version
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If you want, this project can also be packaged with a `setup.bat` file so the user only needs to double-click it and it will create the environment, install dependencies, and start the app automatically.
