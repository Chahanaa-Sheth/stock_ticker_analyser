import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(ticker):
    # Download historical stock data
    df = yf.download(ticker, period="5y")
    df = df.dropna()

    # Save a copy before scaling (used for plotting)
    raw_df = df.copy()

    # Add next-day prediction target
    df["Prediction"] = df["Close"].shift(-1)

    # Select features for ML
    features = ["Open", "High", "Low", "Close", "Volume"]
    processed_df = df[features].copy()

    # Scale the features for model input
    scaler = MinMaxScaler()
    processed_df[features] = scaler.fit_transform(processed_df[features])

    return raw_df, processed_df
