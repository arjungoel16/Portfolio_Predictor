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
    data['Upper_BB'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_BB'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
    data['Support'] = data['Close'].rolling(window=50).min()
    data['Resistance'] = data['Close'].rolling(window=50).max()
    data = data.dropna()
    return data

def prepare_data(data):
    data = add_features(data)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.dropna()

def plot_stock_data(category, stocks_data):
    plt.figure(figsize=(12, 6))
    for ticker, data in stocks_data.items():
        plt.plot(data.index, data['Close'], label=f'{ticker} Close')
        plt.plot(data.index, data['10_day_MA'], linestyle='dashed', label=f'{ticker} 10-Day MA')
        plt.plot(data.index, data['50_day_MA'], linestyle='dotted', label=f'{ticker} 50-Day MA')
        plt.plot(data.index, data['Upper_BB'], linestyle='dashdot', color='gray', label=f'{ticker} Upper BB')
        plt.plot(data.index, data['Lower_BB'], linestyle='dashdot', color='gray', label=f'{ticker} Lower BB')
        plt.plot(data.index, data['Support'], linestyle='solid', color='green', label=f'{ticker} Support')
        plt.plot(data.index, data['Resistance'], linestyle='solid', color='red', label=f'{ticker} Resistance')
    plt.legend()
    plt.title(f"{category} Stocks - Price, Indicators, and Support/Resistance")
    plt.show()

def main():
    categories = {
        "Tech": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
        "Finance": ["GS", "MS", "BAC", "WFC", "JPM", "C", "USB", "PNC", "TFC", "BK"],
        "Indexes": ["^DJI", "^GSPC", "^IXIC"]
    }
    
    stocks_data = {cat: {} for cat in categories}
    
    for category, tickers in categories.items():
        for ticker in tickers:
            print(f"Processing {ticker} ({category})...")
            data = get_stock_data(ticker)
            data = prepare_data(data)
            stocks_data[category][ticker] = data
    
    for category, stocks in stocks_data.items():
        plot_stock_data(category, stocks)

if __name__ == "__main__":
    main()
