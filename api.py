from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

from train_model import train_model  # ⬅️ PENTING

app = Flask(__name__)

MODEL_PATH = "model_telur.pkl"


# =========================
# Helper: load model
# =========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# =========================
# TRAIN MODEL (DARI LARAVEL)
# =========================
@app.route("/train", methods=["POST"])
def train():
    req = request.get_json()

    if not req:
        return jsonify({"error": "Request JSON kosong"}), 400

    dataset = req.get("dataset")
    training = req.get("training")

    if not dataset or not training:
        return jsonify({"error": "Dataset atau training parameter tidak lengkap"}), 400

    try:
        result = train_model(dataset, training)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# PREDICT (MODEL TERBARU)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model belum ditraining"}), 400

    data = request.get_json()

    try:
        fitur = np.array([[
            int(data["jumlah_ayam"]),
            float(data["pakan_total_kg"]),
            int(data["kematian"]),
            int(data["afkir"])
        ]])
    except Exception:
        return jsonify({"error": "Parameter tidak valid"}), 400

    prediksi = model.predict(fitur)

    return jsonify({
        "prediksi_telur_kg": round(float(prediksi[0]), 2)
    })


# =========================
# PREDICTION ALL REQUEST (MODEL TERBARU)
# =========================
@app.route("/train_predict", methods=["POST"])
def train_predict():
    req = request.get_json()

    dataset = req.get("dataset")
    training = req.get("training")
    predict_input = req.get("predict")  # fitur untuk prediksi

    if not dataset or not training or not predict_input:
        return jsonify({
            "status": "error",
            "message": "dataset / training / predict tidak lengkap"
        }), 400

    # TRAINING
    result = train_model(dataset, training)

    # LOAD MODEL TERBARU
    model = load_model()

    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model belum tersedia"
        }), 400

    # PREDIKSI
    try:
        fitur = np.array([[
            float(predict_input["jumlah_ayam"]),
            float(predict_input["pakan_total_kg"]),
            float(predict_input["kematian"]),
            float(predict_input["afkir"])
        ]])
    except Exception:
        return jsonify({"status": "error", "message": "Predict input tidak valid"}), 400

    prediksi = model.predict(fitur)

    return jsonify({
        "status": "success",
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "R2": result["R2"],
        "prediksi_telur_kg": round(float(prediksi[0]), 2)
    })


# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return "🚀 API Random Forest Telur Aktif"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
