import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def get_stock_data(ticker, period="10y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            logging.warning(f"No data found for {ticker}.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_outlook(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Current Price": info.get("currentPrice", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Earnings Growth": info.get("earningsGrowth", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "Health": "Bullish" if info.get("earningsGrowth", 0) > 0 else "Bearish"
        }
    except Exception as e:
        logging.error(f"Error fetching outlook for {ticker}: {e}")
        return {}

def add_features(data):
    if data.empty:
        return data
    
    data = data.copy()
    data['10_day_MA'] = data['Close'].rolling(10).mean()
    data['50_day_MA'] = data['Close'].rolling(50).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].diff().clip(lower=0).rolling(14).mean() /
                               data['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['Upper_BB'] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
    data['Lower_BB'] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
    data['Support'] = data['Close'].rolling(50).min()
    data['Resistance'] = data['Close'].rolling(50).max()
    return data.dropna()

def prepare_data(data):
    if data.empty:
        return data
    
    data = add_features(data)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.dropna()

def train_model(data):
    if data.empty:
        return None
    
    features = ["10_day_MA", "50_day_MA", "RSI", "Volatility", "Upper_BB", "Lower_BB", "Support", "Resistance"]
    X = data[features]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    scores = {
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, predictions)
    }
    logging.info(f"Model Performance: {scores}")
    return model

def forecast_stock(ticker):
    outlook = get_stock_outlook(ticker)
    print(f"Stock Outlook for {ticker}:")
    for key, value in outlook.items():
        print(f"{key}: {value}")

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]  # Example tickers
    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = get_stock_data(ticker)
        data = prepare_data(data)
        if not data.empty:
            model = train_model(data)
            forecast_stock(ticker)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
