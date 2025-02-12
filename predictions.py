import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def get_stock_data(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def add_features(data):
    data = data.copy()
    data['10_day_MA'] = data['Close'].rolling(window=10).mean()
    data['50_day_MA'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                     data['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
    data['Volatility'] = data['Close'].pct_change().rolling(window=10).std()
    data = data.dropna()
    return data

def prepare_data(data):
    data = add_features(data)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.dropna()

def train_model(data):
    predictors = ["Open", "High", "Low", "Close", "Volume", "10_day_MA", "50_day_MA", "RSI", "Volatility"]
    
    if len(data) < 200:
        print("Warning: Not enough data to train the model reliably.")
        return None
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    train[predictors] = scaler.fit_transform(train[predictors])
    test[predictors] = scaler.transform(test[predictors])
    
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=5, random_state=42)
    model.fit(train[predictors], train["Target"])
    predictions = model.predict(test[predictors])
    
    print("Precision Score:", precision_score(test["Target"], predictions, zero_division=0))
    print("Recall Score:", recall_score(test["Target"], predictions))
    print("F1 Score:", f1_score(test["Target"], predictions))
    print("ROC-AUC Score:", roc_auc_score(test["Target"], predictions))
    
    return model

def predict_future(model, data):
    if model is None:
        return None
    predictors = ["Open", "High", "Low", "Close", "Volume", "10_day_MA", "50_day_MA", "RSI", "Volatility"]
    latest_data = data.iloc[-1:]
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
