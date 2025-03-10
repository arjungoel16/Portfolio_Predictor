import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
            "Day's Range": f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            "52 Week Range": f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            "Volume": info.get("volume", "N/A"),
            "Avg Volume": info.get("averageVolume", "N/A"),
            "Market Cap (Intraday)": info.get("marketCap", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Earnings Growth": info.get("earningsGrowth", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Health": "Bullish" if info.get("earningsGrowth", 0) > 0 else "Bearish"
        }
    except Exception as e:
        logging.error(f"Error fetching outlook for {ticker}: {e}")
        return {}

def plot_stock_data(data, ticker):
    st.write("### Stock Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.plot(data.index, data['Close'].rolling(10).mean(), label='10-day MA', color='green')
    ax.plot(data.index, data['Close'].rolling(50).mean(), label='50-day MA', color='red')
    ax.fill_between(data.index, data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std(),
                    data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std(),
                    color='gray', alpha=0.3, label='Bollinger Bands')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title(f'{ticker} Stock Price Chart')
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    
    if st.button("Advanced Chart"):
        st.write("### Advanced Stock Chart")
        st.write("Select comparisons and indicators:")
        
        comparisons = st.multiselect("Compare with:", ["DOW", "SP500", "Nasdaq"], [])
        indicators = st.multiselect("Add indicators:", ["Moving Average", "Bollinger Bands", "RSI", "MACD", "Volume", "VWAP"], [])
        
        for comp in comparisons:
            st.write(f"Comparing with: {comp}")
        for ind in indicators:
            st.write(f"Adding indicator: {ind}")

def forecast_stock(ticker):
    data = get_stock_data(ticker)
    if data.empty:
        st.write("No data available for forecasting.")
        return
    
    data['Tomorrow'] = data['Close'].shift(-1)
    data = data.dropna()
    
    features = ['Open', 'High', 'Low', 'Volume']
    X = data[features]
    y = data['Tomorrow']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    
    predicted_price = model.predict([X.iloc[-1]])[0]
    
    st.write(f"Predicted Closing Price for Tomorrow: ${predicted_price:.2f}")

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
