# EmoSense — Multimodal Emotion Detection

Detect human emotions from **images** (upload / URL / webcam) and **audio** (upload / live recording) using AI.

---

## Project Structure

```
emotion-app/
├── app.py            ← Flask backend (API)
├── index.html        ← Frontend (open in browser)
├── requirements.txt  ← Python dependencies
├── start.bat         ← Windows one-click launch
├── start.sh          ← Mac/Linux one-click launch
└── README.md
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** DeepFace will automatically download model weights (~100 MB) on first run.

### 2. Run the Flask backend

```bash
python app.py
```

You should see:
```
* Running on http://0.0.0.0:5000
```

### 3. Open the frontend

Open `index.html` directly in your browser — no server needed.

---

## Features

| Feature | Method |
|---|---|
| Image from file | Upload JPG/PNG/WEBP |
| Image from URL | Paste any image link |
| Image from webcam | Live capture |
| Audio from file | Upload WAV/MP3/OGG |
| Audio live record | Microphone via browser |

---

## API Endpoints

| Endpoint | Method | Body |
|---|---|---|
| `GET /health` | — | Health check |
| `POST /api/detect/image` | multipart `file` | Upload image |
| `POST /api/detect/image` | JSON `{"url":"..."}` | Image URL |
| `POST /api/detect/image` | JSON `{"image":"base64..."}` | Webcam capture |
| `POST /api/detect/audio` | multipart `file` | Upload/recorded audio |

---

## Troubleshooting

- **"API offline"** — Make sure `python app.py` is running first.
- **"No face detected"** — Use a clear, well-lit photo with a visible face.
- **Camera/mic blocked** — Allow browser permissions for localhost.
- **DeepFace slow on first run** — It downloads models once; subsequent runs are fast.
