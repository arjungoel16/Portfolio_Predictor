import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# machine learning model
from sklearn.metrics import precision_score
#measures model accuracy
from datetime import datetime, timedelta
# handles data related operations

def get_stock_data(ticker, period="max"):
    # max -> get all available data
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def prepare_data(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    # target column -> 1 if stock goes up, 0 if stock goes down
    # removes rows with missing values
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.dropna()

def train_model(data):
    # use random forest classifier to predict stock price movement
    # splits data into training and testing sets
    # evaluates model using precision score
    predictors = ["Open", "High", "Low", "Close", "Volume"]
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(train[predictors], train["Target"])
    predictions = model.predict(test[predictors])
    print("Precision Score:", precision_score(test["Target"], predictions))
    return model

def predict_future(model, data):
    latest_data = data.iloc[-1:]
    predictors = ["Open", "High", "Low", "Close", "Volume"]
    future_prediction = model.predict(latest_data[predictors])
    return future_prediction[0]

def main():
    tech_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
    finance_tickers = ["GS", "MS", "BAC", "WFC", "JPM", "C", "USB", "PNC", "TFC", "BK"]
    index_tickers = ["^DJI", "^GSPC", "^IXIC"]  # Dow Jones, S&P 500, Nasdaq

    models = {}
    predictions = {"Tech": {}, "Finance": {}, "Indexes": {}}

    for ticker in tech_tickers:
        print(f"Processing {ticker} (Tech)...")
        # get stock data, prepare data, train model, predict future
        data = get_stock_data(ticker)
        data = prepare_data(data)
        model = train_model(data)
        prediction = predict_future(model, data)
        models[ticker] = model
        predictions["Tech"][ticker] = prediction

    for ticker in finance_tickers:
        print(f"Processing {ticker} (Finance)...")
        data = get_stock_data(ticker)
        data = prepare_data(data)
        model = train_model(data)
        # train model, predict future bank stock prices
        prediction = predict_future(model, data)
        models[ticker] = model
        predictions["Finance"][ticker] = prediction

    for ticker in index_tickers:
        print(f"Processing {ticker} (Index)...")
        data = get_stock_data(ticker)
        data = prepare_data(data)
        model = train_model(data)
        #same logic as above, but for indexes
        prediction = predict_future(model, data)
        predictions["Indexes"][ticker] = prediction

    # Plot industry comparisons
    tech_up = sum(predictions["Tech"].values())
    finance_up = sum(predictions["Finance"].values())
    # counts the number of stocks predicted to go up in each industry
    plt.bar(["Tech", "Finance"], [tech_up, finance_up], color=["blue", "green"])
    plt.title("Tech vs. Finance - Predictions for Next Market Day")
    plt.ylabel("Number of Stocks Predicted Up")
    plt.show()

    # Compare industries to indexes
    index_up = sum(predictions["Indexes"].values())
    # compare tech, finance, and indexes
    plt.bar(["Tech", "Finance", "Indexes"], [tech_up, finance_up, index_up], color=["blue", "green", "red"])
    plt.title("Industry vs. Indexes - Predictions for Next Market Day")
    plt.ylabel("Number of Stocks/Indexes Predicted Up")
    plt.show()

    print("Predictions for the next market day:")
    for category, category_predictions in predictions.items():
        for ticker, prediction in category_predictions.items():
            print(f"{ticker} ({category}): {'Up' if prediction == 1 else 'Down'}")

if __name__ == "__main__":
    main()
