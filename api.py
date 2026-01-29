from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Request harus JSON"}), 400

    dataset = data.get("dataset")
    training = data.get("training")

    # cek dataset
    if not dataset or len(dataset) < 2:
        return jsonify({
            "status": "error",
            "message": "Dataset minimal 2 baris, sekarang: " + str(len(dataset) if dataset else 0)
        }), 400

    try:
        df = pd.DataFrame(dataset)

        required_cols = ["pakan_total_kg", "kematian", "afkir", "telur_kg"]
        for c in required_cols:
            if c not in df.columns:
                return jsonify({"status": "error", "message": f"Kolom {c} tidak ada"}), 400

        df["pakan_total_kg"] = df["pakan_total_kg"].astype(float)
        df["kematian"] = df["kematian"].astype(int)
        df["afkir"] = df["afkir"].astype(int)
        df["telur_kg"] = df["telur_kg"].astype(float)

        X = df[["pakan_total_kg", "kematian", "afkir"]]
        y = df["telur_kg"]

        n_estimators = int(training.get("n_estimators", 100))
        random_state = int(training.get("random_state", 42))
        max_depth = training.get("max_depth")
        max_depth = int(max_depth) if max_depth else None

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth
        )

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))   # FIX buat sklearn lama
        r2 = r2_score(y_test, pred)

        return jsonify({
            "status": "success",
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 2)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
