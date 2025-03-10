import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st

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
            "Previous Close": info.get("previousClose", "N/A"),
            "Open Price": info.get("open", "N/A"),
            "Bid Price": info.get("bid", "N/A"),
            "Ask Price": info.get("ask", "N/A"),
            "Day's Range": info.get("dayLow", "N/A"),
            "52 Week Range": info.get("fiftyTwoWeekLow", "N/A"),
            "Volume": info.get("volume", "N/A"),
            "Avg Volume": info.get("averageVolume", "N/A"),
            "Market Cap (Intraday)": info.get("marketCap", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Earnings Growth": info.get("earningsGrowth", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "Health": "Bullish" if info.get("earningsGrowth", 0) > 0 else "Bearish"
        }
    except Exception as e:
        logging.error(f"Error fetching outlook for {ticker}: {e}")
        return {}

def plot_stock_data(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['Close'].rolling(10).mean(), label='10-day MA', color='green')
    plt.plot(data.index, data['Close'].rolling(50).mean(), label='50-day MA', color='red')
    plt.fill_between(data.index, data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std(),
                     data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std(),
                     color='gray', alpha=0.3, label='Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{ticker} Stock Price Chart')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

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
    return model

def forecast_stock(ticker):
    data = get_stock_data(ticker)
    if data.empty:
        st.write("No data available for forecasting.")
        return

    data = prepare_data(data)
    model = train_model(data)
    if model:
        st.write(f"Forecast for {ticker}: Model trained successfully!")
    else:
        st.write("Failed to train model.")

def prepare_data(data):
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
    data['Tomorrow'] = data['Close'].shift(-1)
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
    return data.dropna()

def main():
    st.title("Stock Market Analysis")
    ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL):", "NVDA")
    data = get_stock_data(ticker)
    
    if not data.empty:
        plot_stock_data(data, ticker)
        outlook = get_stock_outlook(ticker)
        
        st.write("### Financial Metrics")
        for key, value in outlook.items():
            st.write(f"**{key}:** {value}")
        
        if st.button("Forecast Stock Price with AI"):
            forecast_stock(ticker)
    else:
        st.write("No data found for the given ticker.")

if __name__ == "__main__":
    main()
