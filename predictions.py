import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_stock_data(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def get_stock_outlook(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    outlook = {
        "Current Price": info.get("currentPrice", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "PE Ratio": info.get("trailingPE", "N/A"),
        "Earnings Growth": info.get("earningsGrowth", "N/A"),
        "Revenue Growth": info.get("revenueGrowth", "N/A"),
        "Health": "Bullish" if info.get("earningsGrowth", 0) > 0 else "Bearish"
    }
    return outlook

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

def forecast_stock(ticker):
    outlook = get_stock_outlook(ticker)
    forecast_text = f"Stock Outlook for {ticker}:\n"
    for key, value in outlook.items():
        forecast_text += f"{key}: {value}\n"
    print(forecast_text)

def prepare_data(data):
    data = add_features(data)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.dropna()

def plot_stock_data(ticker, data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1.5)
    
    def update_plot(show_ma, show_bb, show_sr):
        ax.clear()
        ax.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1.5)
        if show_ma:
            ax.plot(data.index, data['10_day_MA'], linestyle='dashed', color='blue', linewidth=2, label='10-Day MA')
            ax.plot(data.index, data['50_day_MA'], linestyle='dotted', color='red', linewidth=2, label='50-Day MA')
        if show_bb:
            ax.fill_between(data.index, data['Upper_BB'], data['Lower_BB'], color='gray', alpha=0.4, label='Bollinger Bands')
        if show_sr:
            ax.plot(data.index, data['Support'], linestyle='solid', color='green', linewidth=2, label='Support')
            ax.plot(data.index, data['Resistance'], linestyle='solid', color='darkred', linewidth=2, label='Resistance')
        ax.set_title(f"{ticker} - Price & Indicators", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.draw()
    
    show_ma = widgets.Checkbox(value=True, description="Moving Averages")
    show_bb = widgets.Checkbox(value=True, description="Bollinger Bands")
    show_sr = widgets.Checkbox(value=True, description="Support/Resistance")
    forecast_button = widgets.Button(description="Generate Forecast")
    forecast_button.on_click(lambda x: forecast_stock(ticker))
    display(show_ma, show_bb, show_sr, forecast_button)
    update_plot(True, True, True)
    plt.show()

def main():
    categories = {
        "Tech": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
        "Finance": ["GS", "MS", "BAC", "WFC", "JPM", "C", "USB", "PNC", "TFC", "BK"],
        "Indexes": ["^DJI", "^GSPC", "^IXIC"]
    }
    
    for category, tickers in categories.items():
        for ticker in tickers:
            print(f"Processing {ticker} ({category})...")
            data = get_stock_data(ticker)
            data = prepare_data(data)
            plot_stock_data(ticker, data)

if __name__ == "__main__":
    main()
