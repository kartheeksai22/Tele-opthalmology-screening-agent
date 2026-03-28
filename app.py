import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
import torch
import cv2
import numpy as np
import base64
from io import BytesIO
from transformers import ViTImageProcessor, AutoModelForImageClassification, pipeline

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP (SQLite)
# ─────────────────────────────────────────────────────────────────────────────
DATABASE = os.path.join(os.path.dirname(__file__), "patients.db")

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DATABASE)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS patient_profile (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT,
            age           REAL,
            gender        TEXT,
            dur           REAL,
            hba1c         REAL,
            fast          REAL,
            pp            REAL,
            bps           REAL,
            bpd           REAL,
            chol          REAL,
            height        REAL,
            weight        REAL,
            smoke         TEXT,
            meds          TEXT,
            dr_stage      TEXT,
            screen_date   TEXT,
            vision        TEXT,
            updated_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date   TEXT NOT NULL UNIQUE,
            fast       REAL,
            pp         REAL,
            water      INTEGER,
            activity   INTEGER,
            meds_taken INTEGER,
            diet       TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS scan_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date  TEXT,
            stage      TEXT,
            confidence TEXT,
            heatmap    TEXT,
            created_at TEXT
        );
    """)
    db.commit()
    db.close()

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT PROFILE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/patient/profile", methods=["GET"])
def get_profile():
    db = get_db()
    row = db.execute("SELECT * FROM patient_profile ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return jsonify({}), 200
    return jsonify(dict(row))

@app.route("/patient/profile", methods=["POST"])
def save_profile():
    data = request.get_json(force=True)
    db = get_db()
    existing = db.execute("SELECT id FROM patient_profile ORDER BY id DESC LIMIT 1").fetchone()
    now = datetime.utcnow().isoformat()
    vision_str = json.dumps(data.get("vision", []))
    fields = (
        data.get("name"), data.get("age"), data.get("gender"),
        data.get("dur"), data.get("hba1c"), data.get("fast"),
        data.get("pp"), data.get("bps"), data.get("bpd"),
        data.get("chol"), data.get("height"), data.get("weight"),
        data.get("smoke"), data.get("meds"), data.get("dr"),
        data.get("screenDate"), vision_str, now
    )
    if existing:
        db.execute("""UPDATE patient_profile SET
            name=?,age=?,gender=?,dur=?,hba1c=?,fast=?,pp=?,bps=?,bpd=?,
            chol=?,height=?,weight=?,smoke=?,meds=?,dr_stage=?,screen_date=?,
            vision=?,updated_at=? WHERE id=?""",
            fields + (existing["id"],))
    else:
        db.execute("""INSERT INTO patient_profile
            (name,age,gender,dur,hba1c,fast,pp,bps,bpd,chol,height,weight,
             smoke,meds,dr_stage,screen_date,vision,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", fields)
    db.commit()
    return jsonify({"status": "ok"})

# ─────────────────────────────────────────────────────────────────────────────
# DAILY LOG ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/patient/daily-logs", methods=["GET"])
def get_daily_logs():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM daily_logs ORDER BY log_date DESC LIMIT 30"
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/patient/daily-log", methods=["POST"])
def save_daily_log():
    data = request.get_json(force=True)
    db = get_db()
    now = datetime.utcnow().isoformat()
    log_date = data.get("date", datetime.utcnow().date().isoformat())
    # Upsert (replace if same date logged again)
    db.execute("""INSERT INTO daily_logs
        (log_date,fast,pp,water,activity,meds_taken,diet,created_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(log_date) DO UPDATE SET
        fast=excluded.fast, pp=excluded.pp, water=excluded.water,
        activity=excluded.activity, meds_taken=excluded.meds_taken,
        diet=excluded.diet""",
        (log_date, data.get("fast"), data.get("pp"),
         data.get("water"), data.get("activity"),
         1 if data.get("meds") else 0,
         data.get("diet"), now))
    db.commit()
    return jsonify({"status": "ok"})

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS ENDPOINT (aggregated trends)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/patient/analytics", methods=["GET"])
def get_analytics():
    db = get_db()
    logs = db.execute(
        "SELECT * FROM daily_logs ORDER BY log_date ASC"
    ).fetchall()
    logs_list = [dict(r) for r in logs]

    if not logs_list:
        return jsonify({"avg_fast": None, "avg_pp": None,
                        "avg_water": None, "streak": 0,
                        "high_sugar_days": 0, "logs": []})

    avg_fast  = round(sum(l["fast"] or 0 for l in logs_list) / len(logs_list), 1)
    avg_pp    = round(sum(l["pp"] or 0 for l in logs_list) / len(logs_list), 1)
    avg_water = round(sum(l["water"] or 0 for l in logs_list) / len(logs_list), 1)
    high_sugar = sum(1 for l in logs_list if (l["fast"] or 0) > 200)

    # Compute current streak
    from datetime import date, timedelta
    today = date.today()
    streak = 0
    dates = sorted({l["log_date"] for l in logs_list}, reverse=True)
    for i, d in enumerate(dates):
        expected = (today - timedelta(days=i)).isoformat()
        if d == expected:
            streak += 1
        else:
            break

    return jsonify({
        "avg_fast": avg_fast,
        "avg_pp": avg_pp,
        "avg_water": avg_water,
        "streak": streak,
        "high_sugar_days": high_sugar,
        "total_logs": len(logs_list),
        "logs": logs_list[-30:]
    })

# ─────────────────────────────────────────────────────────────────────────────
# SCAN HISTORY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/patient/scan-history", methods=["GET"])
def get_scan_history():
    db = get_db()
    rows = db.execute(
        "SELECT id,scan_date,stage,confidence,heatmap FROM scan_history ORDER BY id DESC LIMIT 20"
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/patient/scan-history", methods=["POST"])
def save_scan():
    data = request.get_json(force=True)
    db = get_db()
    now = datetime.utcnow().isoformat()
    db.execute("""INSERT INTO scan_history (scan_date,stage,confidence,heatmap,created_at)
                  VALUES (?,?,?,?,?)""",
               (data.get("scan_date", now), data.get("stage"),
                data.get("confidence"), data.get("heatmap"), now))
    db.commit()
    return jsonify({"status": "ok"})

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL AI MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("Loading models... This may take a moment.")
model_name = "Kontawat/vit-diabetic-retinopathy-classification"
model = AutoModelForImageClassification.from_pretrained(model_name, output_attentions=True)
try:
    processor = ViTImageProcessor.from_pretrained(model_name)
except:
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
gatekeeper = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print("Models loaded successfully!")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENDPOINT (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    try:
        image = Image.open(file).convert("RGB")
    except:
        return jsonify({"error": "Invalid image format"}), 400

    RETINAL_LABELS = [
        "a medical ophthalmology fundus retinal scan showing optic disc and blood vessels",
        "a photograph of a person, face, body, or selfie",
        "a photograph of an outdoor scene, building, street, vehicle, or landscape",
        "a photograph of food, drink, object, or household item",
        "a photograph of an animal or pet",
        "a blurry or dark photograph of miscellaneous objects",
    ]
    gk_result = gatekeeper(image, candidate_labels=RETINAL_LABELS)
    top_prediction = gk_result[0]
    fundus_score = next((r["score"] for r in gk_result if "fundus retinal scan" in r["label"]), 0.0)
    if "fundus retinal scan" not in top_prediction["label"] or fundus_score < 0.70:
        return jsonify({"error": f"Invalid Image: This does not appear to be a retinal fundus scan (confidence: {fundus_score:.0%}). Please submit a proper eye fundus image captured via a macro or 20D lens."}), 400

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence = probabilities[predicted_class_idx].item()

        attentions = outputs.attentions
        last_layer_att = attentions[-1][0]  # (n_heads, 197, 197)

        # ── Select the 4 most spatially-focused heads (highest per-head variance)
        att_np   = last_layer_att.detach().cpu().numpy()
        cls_atts = att_np[:, 0, 1:]           # (n_heads, 196)  — CLS→patch weights
        head_var = cls_atts.var(axis=1)        # variance per head
        top_heads = np.argsort(head_var)[-4:]  # 4 most focused heads
        mean_cls  = cls_atts[top_heads].mean(axis=0)  # (196,)

        heatmap_grid = mean_cls.reshape(14, 14)

        # ── Normalise
        hm_min, hm_max = heatmap_grid.min(), heatmap_grid.max()
        heatmap_grid = (heatmap_grid - hm_min) / (hm_max - hm_min + 1e-8)

        # ── Keep only the TOP 10% attention (90th-percentile threshold) — precise spots only
        thr = float(np.percentile(heatmap_grid, 90))
        heatmap_focused = np.where(heatmap_grid > thr, heatmap_grid - thr, 0.0)
        if heatmap_focused.max() > 0:
            heatmap_focused /= heatmap_focused.max()          # re-normalise to [0, 1]
            heatmap_focused = np.power(heatmap_focused, 1.5)  # gamma: concentrate peak spots, suppress edges

        # ── Resize to original image size
        orig_w, orig_h = image.size
        heatmap_resized = cv2.resize(heatmap_focused.astype(np.float32), (orig_w, orig_h))

        # ── Tight Gaussian blur (7×7) for smooth but small spot edges
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
        if heatmap_resized.max() > 0:
            heatmap_resized /= heatmap_resized.max()

        # ── Hard mask: zero out anything below 15% intensity — keeps overlay only on real lesion spots
        hard_mask = (heatmap_resized > 0.15).astype(np.float32)
        heatmap_masked = heatmap_resized * hard_mask

        # ── Use JET colormap (blue→green→yellow→red) — clear medical visualization of diseased spots
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_masked), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32)

        # ── Per-pixel alpha blend: original untouched where no attention (lighter 0.55 blend)
        alpha = heatmap_masked[:, :, np.newaxis]      # (H, W, 1)
        orig_f = np.array(image.convert("RGB")).astype(np.float32)
        blended = orig_f * (1 - alpha * 0.55) + heatmap_color * (alpha * 0.55)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended_pil = Image.fromarray(blended)
        buffered = BytesIO()
        blended_pil.save(buffered, format="JPEG", quality=85)
        heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    stage = labels[predicted_class_idx]
    conf_str = f"{confidence:.2%}"

    # Auto-save scan to DB
    db = get_db()
    now = datetime.utcnow().isoformat()
    db.execute("""INSERT INTO scan_history (scan_date,stage,confidence,heatmap,created_at)
                  VALUES (?,?,?,?,?)""",
               (now, stage, conf_str, heatmap_b64, now))
    db.commit()

    return jsonify({"severity": stage, "confidence": conf_str, "heatmap": heatmap_b64})

# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENDPOINT (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
METRICS_FILE = "performance_metrics.json"

@app.route("/metrics", methods=["GET"])
def get_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found. Run training first."}), 404

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)