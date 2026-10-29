

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import tempfile
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from bson import ObjectId
print("=" * 50)
print("RUNNING APP FROM:", __file__)
print("GROQ_API_KEY:", os.environ.get("GROQ_API_KEY"))
print("=" * 50)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "emosense"

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    mongo_client.server_info()          # ping — will raise if unreachable
    db              = mongo_client[DB_NAME]
    image_col       = db["image_detections"]
    audio_col       = db["audio_detections"]
    MONGO_CONNECTED = True
    print(f"✅  MongoDB connected → {MONGO_URI} / {DB_NAME}")
except Exception as e:
    MONGO_CONNECTED = False
    db = image_col = audio_col = None
    print(f"⚠️   MongoDB unavailable ({e}). Detections will NOT be saved.")


app = Flask(__name__)
CORS(app)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save(collection, document: dict):
    """Insert a document and return its string _id (or None on failure)."""
    if not MONGO_CONNECTED or collection is None:
        return None
    try:
        result = collection.insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        print(f"MongoDB insert error: {e}")
        return None


def _serialize(doc: dict) -> dict:
    """Convert ObjectId / datetime fields so they JSON-serialise cleanly."""
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    if "timestamp" in out and isinstance(out["timestamp"], datetime):
        out["timestamp"] = out["timestamp"].isoformat()
    return out


# ─── Image Emotion Detection ───────────────────────────────────────────────────

def analyze_image_emotion(image_source, source_type="file"):
    try:
        from deepface import DeepFace
        import cv2

        if source_type == "base64":
            img_data = base64.b64decode(image_source)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "Could not decode image data"
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img)
                image_source = tmp.name
                source_type = "file"

        # Try progressively more lenient detectors so imperfect photos
        # still produce a real emotion rather than defaulting to "fearful".
        last_err = None
        result = None
        for backend in ["retinaface", "mtcnn", "opencv"]:
            try:
                result = DeepFace.analyze(
                    img_path=image_source,
                    actions=["emotion"],
                    enforce_detection=False,   # don't crash on imperfect crops
                    detector_backend=backend,
                    silent=True
                )
                break
            except Exception as e:
                last_err = e
                continue

        if result is None:
            raise last_err or Exception("All detectors failed")

        if isinstance(result, list):
            result = result[0]

        emotions_raw = result["emotion"]

        # ── Fearful bias correction ──────────────────────────────────────
        # DeepFace's default weights systematically over-predict "fear" on
        # neutral/ambiguous faces.  We dampen it unless it truly dominates.
        fear_score = emotions_raw.get("fear", emotions_raw.get("fearful", 0))
        other_max  = max(v for k, v in emotions_raw.items()
                         if k not in ("fear", "fearful"))
        if fear_score < other_max * 1.5:          # fear not a clear winner
            for k in ("fear", "fearful"):
                if k in emotions_raw:
                    emotions_raw[k] *= 0.55       # dampen fear

        # Re-normalise to 100 %
        total_e = sum(emotions_raw.values()) or 1
        emotions = {k: round(v / total_e * 100, 1) for k, v in emotions_raw.items()}
        dominant   = max(emotions, key=emotions.__getitem__)
        confidence = emotions[dominant]

        if source_type == "file" and "tmp" in str(image_source):
            try:
                os.unlink(image_source)
            except:
                pass

        return {
            "dominant_emotion": dominant,
            "confidence": confidence,
            "all_emotions": {k: round(v, 1) for k, v in emotions.items()}
        }, None

    except Exception as e:
        err = str(e)
        if "Face could not be detected" in err or "No face" in err:
            return None, "No face detected in the image. Please use a clear photo with a visible face."
        if "ModuleNotFoundError" in err or "No module named" in err:
            return None, f"Missing dependency: {err}"
        return None, f"Image analysis error: {err}"


# ─── Audio Emotion Detection ───────────────────────────────────────────────────

AUDIO_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


def extract_audio_features(file_path):
    import librosa
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (< 0.5 seconds)")

    mfccs        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean    = np.mean(mfccs, axis=1)
    mfcc_std     = np.std(mfccs,  axis=1)
    chroma       = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean  = np.mean(chroma, axis=1)
    contrast     = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    rms          = librosa.feature.rms(y=y)
    energy       = float(np.mean(rms))
    zcr          = librosa.feature.zero_crossing_rate(y)
    zcr_mean     = float(np.mean(zcr))
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals   = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch    = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    features = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean, contrast_mean,
        [energy, zcr_mean, avg_pitch]
    ])
    return features, y, sr


def predict_audio_emotion(features, y, sr):
    """
    Rule-based emotion scorer using calibrated audio features.

    Key fixes vs. original:
    - mfcc_var > 500 fired on virtually every input (MFCC values are large),
      making "fearful" win almost always.  Replaced with per-feature z-score
      logic and realistic thresholds derived from librosa's typical output ranges.
    - All base scores start at 0 so no emotion gets an unearned head-start.
    - Scores are softmax-normalised for smoother probabilities.
    """
    import math

    energy   = features[-3]   # RMS — typical speech: 0.005–0.08
    zcr      = features[-2]   # zero-crossing rate — typical: 0.02–0.20
    pitch    = features[-1]   # average pitch Hz — typical: 80–400
    mfcc_mean_vals = features[:40]

    # Spectral flux proxy: std of MFCC means (higher = more tonal variation)
    mfcc_spread = float(np.std(mfcc_mean_vals))   # typical: 10–60
    # Brightness: mean of positive MFCC means (higher = brighter/louder)
    mfcc_brightness = float(np.mean([v for v in mfcc_mean_vals if v > 0] or [0]))

    scores = {
        "neutral":   0.0,
        "happy":     0.0,
        "sad":       0.0,
        "angry":     0.0,
        "fearful":   0.0,
        "disgusted": 0.0,
        "surprised": 0.0,
    }

    # ── Happy: high energy, high pitch, moderate-high ZCR ────────────────
    if energy > 0.03 and pitch > 180:
        scores["happy"] += 0.40
    if energy > 0.05 and pitch > 220:
        scores["happy"] += 0.20
    if mfcc_brightness > 20:
        scores["happy"] += 0.10

    # ── Surprised: very high pitch spike OR sudden energy burst ──────────
    if pitch > 300 and energy > 0.04:
        scores["surprised"] += 0.35
    if pitch > 350:
        scores["surprised"] += 0.20

    # ── Angry: high energy, high ZCR, lower/mid pitch ────────────────────
    if energy > 0.05 and zcr > 0.08:
        scores["angry"] += 0.35
    if energy > 0.07 and zcr > 0.12:
        scores["angry"] += 0.25
    if mfcc_spread > 35 and energy > 0.04:
        scores["angry"] += 0.10

    # ── Sad: low energy, low pitch, low ZCR ──────────────────────────────
    if energy < 0.018 and pitch < 160:
        scores["sad"] += 0.40
    if energy < 0.012:
        scores["sad"] += 0.20
    if zcr < 0.04 and energy < 0.025:
        scores["sad"] += 0.10

    # ── Fearful: low-moderate energy, high ZCR, tremor (high mfcc_spread)
    # Crucially: only fires when ZCR is genuinely high AND energy is LOW
    if energy < 0.025 and zcr > 0.10 and mfcc_spread > 30:
        scores["fearful"] += 0.35
    if 0.008 < energy < 0.02 and zcr > 0.13:
        scores["fearful"] += 0.20

    # ── Disgusted: low pitch, high ZCR irregularity ──────────────────────
    if zcr > 0.14 and pitch < 150:
        scores["disgusted"] += 0.30
    if mfcc_spread < 15 and energy > 0.02 and pitch < 130:
        scores["disgusted"] += 0.15

    # ── Neutral: mid-range everything — the "default" when nothing stands out
    if 0.012 <= energy <= 0.045 and 0.03 <= zcr <= 0.10 and 100 <= pitch <= 250:
        scores["neutral"] += 0.35
    if 0.018 <= energy <= 0.035:
        scores["neutral"] += 0.15

    # ── Softmax normalisation (temperature=1.5 for smoother distribution) ──
    temperature = 1.5
    vals = np.array(list(scores.values())) / temperature
    exp_vals = np.exp(vals - np.max(vals))      # numerical stability
    softmax  = exp_vals / exp_vals.sum()

    scores_pct = {k: round(float(v) * 100, 1) for k, v in zip(scores.keys(), softmax)}
    dominant   = max(scores_pct, key=scores_pct.__getitem__)
    return dominant, scores_pct


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Emotion Detection API running",
        "mongo": MONGO_CONNECTED
    })


@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    try:
        data   = request.get_json(silent=True)
        source = "unknown"

        # ── URL input ──
        if data and data.get("url"):
            import urllib.request
            url = data["url"]
            source = "url"
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                result, error = analyze_image_emotion(tmp.name, "file")
            try:
                os.unlink(tmp.name)
            except:
                pass

        # ── Base64 (webcam) ──
        elif data and data.get("image"):
            b64 = data["image"]
            source = "webcam"
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            result, error = analyze_image_emotion(b64, "base64")

        # ── File upload ──
        elif "file" in request.files:
            file = request.files["file"]
            source = "upload"
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400
            allowed = {"jpg", "jpeg", "png", "bmp", "webp"}
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in allowed:
                return jsonify({"success": False, "error": f"Unsupported format. Use: {', '.join(allowed)}"}), 400
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                file.save(tmp.name)
                result, error = analyze_image_emotion(tmp.name, "file")
            try:
                os.unlink(tmp.name)
            except:
                pass

        else:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        if error:
            return jsonify({"success": False, "error": error}), 400

        # ── Persist to MongoDB ──
        doc = {
            "type":             "image",
            "source":           source,
            "dominant_emotion": result["dominant_emotion"],
            "confidence":       result["confidence"],
            "all_emotions":     result["all_emotions"],
            "timestamp":        datetime.now(timezone.utc),
        }
        saved_id = _save(image_col, doc)

        return jsonify({"success": True, "result": result, "saved_id": saved_id})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/detect/audio", methods=["POST"])
def detect_audio():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        allowed = {"wav", "mp3", "ogg", "webm", "m4a", "flac"}
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "wav"
        if ext not in allowed:
            return jsonify({"success": False, "error": f"Unsupported audio format. Use: {', '.join(allowed)}"}), 400

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            features, y, sr = extract_audio_features(tmp_path)
            dominant, scores = predict_audio_emotion(features, y, sr)
        except ValueError as ve:
            os.unlink(tmp_path)
            return jsonify({"success": False, "error": str(ve)}), 400
        except Exception as e:
            import traceback
            traceback.print_exc()
            os.unlink(tmp_path)
            return jsonify({"success": False, "error": f"Audio processing error: {str(e)}"}), 500
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

        result = {
            "dominant_emotion": dominant,
            "confidence":       scores[dominant],
            "all_emotions":     scores
        }

        # ── Persist to MongoDB ──
        doc = {
            "type":             "audio",
            "source":           "upload",
            "dominant_emotion": dominant,
            "confidence":       scores[dominant],
            "all_emotions":     scores,
            "timestamp":        datetime.now(timezone.utc),
        }
        saved_id = _save(audio_col, doc)

        return jsonify({"success": True, "result": result, "saved_id": saved_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── History / Stats Routes ────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the last N detections (image + audio, newest first)."""
    if not MONGO_CONNECTED:
        return jsonify({"success": False, "error": "MongoDB not connected"}), 503

    limit     = min(int(request.args.get("limit", 50)), 200)
    type_filter = request.args.get("type")          # "image" | "audio" | None

    records = []
    collections = []
    if type_filter == "image":
        collections = [(image_col, "image")]
    elif type_filter == "audio":
        collections = [(audio_col, "audio")]
    else:
        collections = [(image_col, "image"), (audio_col, "audio")]

    for col, t in collections:
        cursor = col.find({}, {"_id": 1, "dominant_emotion": 1,
                               "confidence": 1, "source": 1, "timestamp": 1}) \
                    .sort("timestamp", -1).limit(limit)
        for doc in cursor:
            doc["type"] = t
            records.append(_serialize(doc))

    # merge-sort by timestamp descending
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return jsonify({"success": True, "records": records[:limit]})


@app.route("/api/history/<record_id>", methods=["GET"])
def get_record(record_id):
    """Return a single detection record by _id (searches both collections)."""
    if not MONGO_CONNECTED:
        return jsonify({"success": False, "error": "MongoDB not connected"}), 503
    try:
        oid = ObjectId(record_id)
    except Exception:
        return jsonify({"success": False, "error": "Invalid ID"}), 400

    for col, t in [(image_col, "image"), (audio_col, "audio")]:
        doc = col.find_one({"_id": oid})
        if doc:
            doc["type"] = t
            return jsonify({"success": True, "record": _serialize(doc)})

    return jsonify({"success": False, "error": "Record not found"}), 404


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    """Delete a single detection record by _id."""
    if not MONGO_CONNECTED:
        return jsonify({"success": False, "error": "MongoDB not connected"}), 503
    try:
        oid = ObjectId(record_id)
    except Exception:
        return jsonify({"success": False, "error": "Invalid ID"}), 400

    for col in [image_col, audio_col]:
        res = col.delete_one({"_id": oid})
        if res.deleted_count:
            return jsonify({"success": True, "deleted_id": record_id})

    return jsonify({"success": False, "error": "Record not found"}), 404


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Aggregate emotion counts across both collections."""
    if not MONGO_CONNECTED:
        return jsonify({"success": False, "error": "MongoDB not connected"}), 503

    def agg(col):
        pipeline = [{"$group": {"_id": "$dominant_emotion", "count": {"$sum": 1}}}]
        return {doc["_id"]: doc["count"] for doc in col.aggregate(pipeline)}

    image_counts = agg(image_col)
    audio_counts = agg(audio_col)

    # combined totals
    all_emotions = set(image_counts) | set(audio_counts)
    combined     = {e: image_counts.get(e, 0) + audio_counts.get(e, 0) for e in all_emotions}

    return jsonify({
        "success": True,
        "stats": {
            "image":    image_counts,
            "audio":    audio_counts,
            "combined": combined,
            "totals": {
                "image": image_col.count_documents({}),
                "audio": audio_col.count_documents({}),
            }
        }
    })


# ─── AI Report Generation ──────────────────────────────────────────────────────

@app.route("/api/report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == "OPTIONS":
        return "", 204
    """
    Generate an emotion insight report using Claude (streamed via SSE).
    Body: { "emotion": str, "confidence": float, "all_emotions": dict, "type": "image"|"audio" }
    """
    try:
        import json as _json

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        emotion      = data.get("emotion", "neutral")
        confidence   = data.get("confidence", 0)
        all_emotions = data.get("all_emotions", {})
        source_type  = data.get("type", "image")

        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
        second = next((e for e in sorted_emotions if e[0] != emotion and e[1] > 15), None)
        blend_context = f" There is also a notable secondary presence of {second[0]} at {second[1]}%." if second else ""
        blend_line    = f"\n**Emotion Blend**\nOne sentence about the secondary emotion ({second[0]}) and what this mix suggests." if second else ""

        if source_type == "audio":
            source_desc = "A voice recording was analysed — emotion detected from pitch, energy, rhythm, and tone"
        else:
            source_desc = "A facial image was analysed — emotion detected from the facial expression"

        prompt = f"""{source_desc}:

- Dominant emotion: {emotion} ({confidence}% confidence)
- Full breakdown: {', '.join(f'{k}: {v}%' for k, v in sorted_emotions)}{blend_context}


Write a short, warm, actionable insight report in exactly this structure:

**Summary**
One sentence: what was detected and what it likely means emotionally. Mention confidence naturally. Warm tone, not clinical.
{blend_line}
**Suggestions**
Exactly 4 specific, practical things the person can do RIGHT NOW for the emotion "{emotion}". Each on its own line starting with →. Make them concrete — e.g. for fear: go somewhere quiet, not just "talk to someone".

Keep total response under 150 words. No # headers. Only **bold** for section labels above."""
        print("=" * 60)
        print("Inside /api/report")
        print("Current working directory:", os.getcwd())
        print("Environment contains GROQ_API_KEY:", "GROQ_API_KEY" in os.environ)
        print("Value:", repr(os.environ.get("GROQ_API_KEY")))
        print("=" * 60)

        api_key = os.environ.get("GROQ_API_KEY")
        print("DEBUG - GROQ_API_KEY:", api_key)
        if not api_key:
            return jsonify({"success": False, "error": "GROQ_API_KEY not set. Run: set GROQ_API_KEY=your-key (Windows) or export GROQ_API_KEY=your-key (Mac/Linux)"}), 500

        import requests as _req

        def stream_report():
            try:
                resp = _req.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 300,
                        "stream": False
                    },
                    timeout=30
                )

                resp.raise_for_status()

                data = resp.json()

                text = data["choices"][0]["message"]["content"]

                yield "data: {\"type\":\"start\"}\n\n"
                yield f"data: {_json.dumps({'type':'text','text':text})}\n\n"
                yield "data: {\"type\":\"done\"}\n\n"

            except Exception as e:
                print(e)
                yield f"data: {_json.dumps({'type':'error','error':str(e)})}\n\n"

        return app.response_class(
            stream_report(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)