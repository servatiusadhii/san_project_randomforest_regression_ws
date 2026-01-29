from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

app = Flask(__name__)

def train_model(dataset, training_params):
    df = pd.DataFrame(dataset)

    # bersihin kolom yang ga dipakai
    df = df[["pakan_total_kg", "kematian", "afkir", "telur_kg"]]

    # convert
    df["pakan_total_kg"] = df["pakan_total_kg"].astype(float)
    df["kematian"] = df["kematian"].astype(int)
    df["afkir"] = df["afkir"].astype(int)
    df["telur_kg"] = df["telur_kg"].astype(float)

    X = df.drop("telur_kg", axis=1)
    y = df["telur_kg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=training_params["random_state"]
    )

    model = RandomForestRegressor(
        n_estimators=training_params["n_estimators"],
        random_state=training_params["random_state"],
        max_depth=training_params["max_depth"]
    )

    model.fit(X_train, y_train)

    # save model
    with open("model_telur.pkl", "wb") as f:
        pickle.dump(model, f)

    # metrics
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)

    return model, mae, rmse, r2

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    dataset = data.get("dataset")
    training_params = data.get("training")
    predict_input = data.get("predict")

    if dataset is None or training_params is None:
        return jsonify({"status": "error", "message": "dataset / training missing"}), 400

    # training
    model, mae, rmse, r2 = train_model(dataset, training_params)

    # prediksi jika ada input
    if predict_input:
        fitur = np.array([[ 
            float(predict_input["pakan_total_kg"]),
            int(predict_input["kematian"]),
            int(predict_input["afkir"])
        ]])

        prediksi = model.predict(fitur)
        prediksi_telur_kg = round(float(prediksi[0]), 2)
    else:
        prediksi_telur_kg = None

    return jsonify({
        "status": "success",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2),
        "prediksi_telur_kg": prediksi_telur_kg
    })


@app.route("/", methods=["GET"])
def home():
    return "API Telur Siap"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
