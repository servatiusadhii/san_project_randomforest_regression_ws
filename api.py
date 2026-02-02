from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Request JSON kosong"}), 400

    dataset = data.get("dataset")
    training = data.get("training", {})

    if not dataset:
        return jsonify({"status": "error", "message": "Dataset tidak ada"}), 400

    if len(dataset) < 10:
        return jsonify({
            "status": "error",
            "message": "Dataset minimal 10 baris"
        }), 400

    try:
        # =====================
        # 1. Dataset → DataFrame
        # =====================
        df = pd.DataFrame(dataset)

        required_cols = [
            "jumlah_ayam",
            "pakan_total_kg",
            "kematian",
            "afkir",
            "telur_kg"
        ]

        for c in required_cols:
            if c not in df.columns:
                return jsonify({
                    "status": "error",
                    "message": f"Kolom '{c}' tidak ditemukan"
                }), 400

        # =====================
        # 2. Feature Engineering (ANTI BOCOR)
        # =====================
        df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]

        X = df[[
            "jumlah_ayam",
            "pakan_per_ayam",
            "kematian",
            "afkir",
        ]]

        y = df["telur_kg"]

        # =====================
        # 3. Training params
        # =====================
        n_estimators = int(training.get("n_estimators", 150))
        random_state = int(training.get("random_state", 42))
        max_depth = training.get("max_depth", 6)

        # =====================
        # 4. Split data
        # =====================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # =====================
        # 5. Model (ANTI OVERFIT)
        # =====================
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # =====================
        # 6. Evaluasi
        # =====================
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        accuracy_pct = 100 - (mae / y_test.mean() * 100)

        # =====================
        # 7. Save model
        # =====================
        with open("model_telur.pkl", "wb") as f:
            pickle.dump(model, f)

        # =====================
        # 8. PREDIKSI (PAKAI DATA TERAKHIR)
        # =====================
        last_row = df.iloc[-1:]

        X_last = last_row[[
            "jumlah_ayam",
            "pakan_per_ayam",
            "kematian",
            "afkir",
        ]]

        pred_harian_kg = float(model.predict(X_last)[0])
        pred_bulanan_kg = pred_harian_kg * 30

        # =====================
        # 9. TELUR PER AYAM (KG)
        # =====================
        avg_jumlah_ayam = df["jumlah_ayam"].mean()
        telur_per_ayam = pred_harian_kg / avg_jumlah_ayam

        # =====================
        # 10. HITUNG BERAT TELUR PER BUTIR (DATA HISTORIS)
        # =====================
        # kg telur per ayam per hari
        df["telur_per_ayam_kg"] = df["telur_kg"] / df["jumlah_ayam"]

        # estimasi berat telur (kg / butir) dari konsistensi data
        avg_telur_per_ayam_kg = df["telur_per_ayam_kg"].mean()
        produksi_rate = df["telur_kg"].sum() / df["jumlah_ayam"].sum()

        berat_telur_kg = avg_telur_per_ayam_kg / produksi_rate

        # safety net (anti NaN / 0)
        if berat_telur_kg <= 0 or np.isnan(berat_telur_kg):
            berat_telur_kg = 0.06  # fallback TERAKHIR (jarang kepake)

        # =====================
        # 11. KONVERSI KE BUTIR
        # =====================
        pred_harian_butir = pred_harian_kg / berat_telur_kg
        pred_bulanan_butir = pred_harian_butir * 30

        # =====================
        # 12. AKURASI PER AYAM
        # =====================
        mae_per_ayam = mae / avg_jumlah_ayam
        mse_per_ayam = mean_squared_error(y_test, y_pred) / (avg_jumlah_ayam ** 2)
        rmse_per_ayam = np.sqrt(mse_per_ayam)

        # =====================
        # 13. RESPONSE FINAL (1:1 DENGAN LARAVEL)
        # =====================
        return jsonify({
            "status": "success",

            "prediksi": {
                "harian_telur_kg": round(pred_harian_kg, 2),
                "bulanan_telur_kg": round(pred_bulanan_kg, 2),
                "telur_per_ayam": round(telur_per_ayam, 4),
                "harian_telur_butir": int(round(pred_harian_butir)),
                "bulanan_telur_butir": int(round(pred_bulanan_butir)),
            },

            "akurasi": {
                "MAE_per_ayam": round(mae_per_ayam, 4),
                "MSE_per_ayam": round(mse_per_ayam, 4),
                "RMSE_per_ayam": round(rmse_per_ayam, 4),
                "R2": round(float(r2), 3),
            }
        })


    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "🚀 API Training Model Produksi Telur (ANTI DATA BOCOR)"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
