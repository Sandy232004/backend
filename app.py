
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = load_model("model/lstm_model_clean.keras")

def get_stock_data(ticker):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365)
    df = yf.download(ticker, start=start, end=end)
    return df

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data["Close"].values.reshape(-1, 1)
    scaled = scaler.fit_transform(close_prices)

    x_test = []
    for i in range(60, len(scaled)):
        x_test.append(scaled[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, scaler

@app.route("/predict", methods=["POST"])
def predict():
    req_data = request.get_json()
    ticker = req_data.get("ticker")

    data = get_stock_data(ticker)
    x_test, scaler = preprocess_data(data)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return jsonify({"predictions": predictions.flatten().tolist()})

if __name__ == "__main__":
    app.run(debug=True)
