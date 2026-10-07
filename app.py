"""Aplikasi web Flask untuk prediksi risiko diabetes (Logistic Regression).

Jalankan:  python app.py   ->  buka http://127.0.0.1:5000
Model dibaca dari folder models/ (hasil notebook: model_diabetes.pkl,
scaler_diabetes.pkl, info_model.json).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
model = joblib.load(BASE / "models" / "model_diabetes.pkl")
scaler = joblib.load(BASE / "models" / "scaler_diabetes.pkl")
info = json.loads((BASE / "models" / "info_model.json").read_text(encoding="utf-8"))
FITUR = info["fitur"]
THRESHOLD = info.get("threshold_default", 0.5)

# Metadata kolom input: dipakai untuk membentuk form dan validasi server
FIELDS = [
    dict(name="pregnancies", label="Jumlah kehamilan", unit="kali", min=0, max=20, step=1, value=2),
    dict(name="glucose", label="Glukosa plasma", unit="mg/dL", min=50, max=250, step=1, value=120),
    dict(name="blood_pressure", label="Tekanan darah diastolik", unit="mmHg", min=30, max=130, step=1, value=70),
    dict(name="bmi", label="Indeks massa tubuh (BMI)", unit="kg/m²", min=10, max=60, step=0.1, value=28),
    dict(name="age", label="Usia", unit="tahun", min=18, max=90, step=1, value=40),
]
LABELS = {f["name"]: f["label"] for f in FIELDS}

app = Flask(__name__)


def kategori(p):
    if p < 0.35:
        return "rendah"
    if p < 0.6:
        return "sedang"
    return "tinggi"


@app.get("/")
def index():
    return render_template("index.html", fields=FIELDS, metrik=info["metrik"])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    nilai = {}
    for f in FIELDS:
        try:
            v = float(data[f["name"]])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"{f['label']} harus berupa angka."), 400
        if not f["min"] <= v <= f["max"]:
            return jsonify(error=f"{f['label']} harus antara {f['min']} dan {f['max']} {f['unit']}."), 400
        nilai[f["name"]] = v

    x = pd.DataFrame([nilai])[FITUR]          # urutan kolom sama dengan saat pelatihan
    z = scaler.transform(x)
    prob = float(model.predict_proba(z)[0, 1])
    kontribusi = [
        {"fitur": LABELS[n], "nilai": float(c)}
        for n, c in zip(FITUR, z[0] * model.coef_[0])
    ]
    kontribusi.sort(key=lambda k: abs(k["nilai"]), reverse=True)
    return jsonify(
        probabilitas=prob,
        prediksi="Diabetes" if prob >= THRESHOLD else "Tidak diabetes",
        kategori=kategori(prob),
        kontribusi=kontribusi,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
