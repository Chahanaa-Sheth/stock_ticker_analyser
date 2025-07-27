# train_model.py

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    data = data.dropna()

    if len(data) < 10:
        return False  # not enough data

    # Feature Engineering
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    X = data[["Open", "High", "Low", "Close", "Volume"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{ticker}_model.pkl")

    return True
