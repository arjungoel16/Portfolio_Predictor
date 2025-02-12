import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import datetime, timedelta

def get_stock_data(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def prepare_data(data):
    data = data.copy()
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.dropna()

def train_model(data):
    predictors = ["Open", "High", "Low", "Close", "Volume"]
    
    if len(data) < 200:
        print("Warning: Not enough data to train the model reliably.")
        return None
    
    train = data.iloc[:-200]
    test = data.iloc[-200:]
    
    if train.empty:
        print("Error: Training set is empty. Model cannot be trained.")
        return None
    
    model = RandomForestClassifier(n_estimators=300, min_samples_split=20, max_depth=10, random_state=42)
    model.fit(train[predictors], train["Target"])
    predictions = model.predict(test[predictors])
    
    print("Precision Score:", precision_score(test["Target"], predictions, zero_division=0))
    return model

def predict_future(model, data):
    if model is None:
        return None
    latest_data = data.iloc[-1:]
    predictors = ["Open", "High", "Low", "Close", "Volume"]
    return model.predict(latest_data[predictors])[0]

def main():
    categories = {
        "Tech": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
        "Finance": ["GS", "MS", "BAC", "WFC", "JPM", "C", "USB", "PNC", "TFC", "BK"],
        "Indexes": ["^DJI", "^GSPC", "^IXIC"]
    }
    
    predictions = {cat: {} for cat in categories}
    
    for category, tickers in categories.items():
        for ticker in tickers:
            print(f"Processing {ticker} ({category})...")
            data = get_stock_data(ticker)
            data = prepare_data(data)
            model = train_model(data)
            prediction = predict_future(model, data)
            if prediction is not None:
                predictions[category][ticker] = prediction
    
    # Graph Comparison
    category_up_counts = {cat: sum(pred.values()) for cat, pred in predictions.items() if pred}
    plt.bar(category_up_counts.keys(), category_up_counts.values(), color=["blue", "green", "red"])
    plt.title("Predictions for Next Market Day")
    plt.ylabel("Number of Stocks Predicted Up")
    plt.show()
    
    # Print Predictions
    print("Predictions for the next market day:")
    for category, category_predictions in predictions.items():
        for ticker, prediction in category_predictions.items():
            print(f"{ticker} ({category}): {'Up' if prediction == 1 else 'Down'}")

if __name__ == "__main__":
    main()
