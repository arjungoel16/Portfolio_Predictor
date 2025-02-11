# Stock Market Prediction System

## Overview
    This project implements a stock market prediction system using machine learning (Random Forest Classifier) to forecast stock movements for two major industries—Technology and Finance—as well as their comparison to major U.S. stock indexes (Dow Jones, S&P 500, and Nasdaq). The system automatically fetches historical data, trains a model, and predicts future stock movements.

## Features
    - Predicts stock movements for:
    - Technology stocks (e.g., NVDA, AAPL, MSFT, etc.)
    - Financial institutions (e.g., GS, MS, BAC, etc.)
    - U.S. stock indexes (Dow Jones, S&P 500, Nasdaq)
    - Utilizes machine learning (Random Forest) to train predictive models
    - Automates data collection from Yahoo Finance
    - Compares industry trends and predictions with visualized bar charts

## Technologies Used
    - Python
    - yFinance (for fetching stock data)
    - pandas, numpy (for data manipulation)
    - scikit-learn (Random Forest Classifier for predictions)
    - matplotlib (for visualizing industry trends)

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-repo/stock-prediction.git
   ```
2. Install dependencies:
   ```
   pip install yfinance pandas numpy matplotlib scikit-learn
   ```
3. Run the script:
   ```
   python predict_stocks.py
   ```

## How It Works
    1. Fetches historical stock data from Yahoo Finance.
    2. Prepares data by adding future price movement labels.
    3. Trains a Random Forest model to predict future movements.
    4. Generates predictions for the next market day.
    5. Compares industry predictions with major stock indexes.
    6. Visualizes trends using bar charts.

## Example Output
```
Processing NVDA (Tech)...
Processing AAPL (Tech)...
Processing GS (Finance)...
Processing ^GSPC (Index)...
...
Predictions for the next market day:
NVDA (Tech): Up
AAPL (Tech): Down
GS (Finance): Up
^GSPC (Index): Up
```

## Visualization
    - Industry comparison graph: Compares the number of stocks predicted to go up in Tech vs. Finance.
    - Market-wide comparison graph: Compares Tech and Finance predictions against U.S. stock indexes.

## Future Improvements
    - Incorporate deep learning models for improved accuracy
    - Add real-time stock movement updates
    - Expand industry coverage beyond Tech & Finance

## Author: Arjun Goel
    Email: arjun.goel6@gmail.com
    LinkedIn: https://www.linkedin.com/in/arjun-goel-vt/

