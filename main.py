# ===============================================
# STOCK PRICE PREDICTION SYSTEM
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
import urllib.request
import ssl
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')

# ===============================================
# STEP 1: LOAD S&P 500 TICKERS (Wikipedia)
# ===============================================

print("Fetching S&P 500 company tickers...")

try:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)

    ssl._create_default_https_context = ssl._create_unverified_context
    tickers = pd.read_html(req)[0]['Symbol']

    print(f"✅ Total Tickers Loaded: {len(tickers)}")
except Exception as e:
    print("⚠️ Could not fetch tickers from Wikipedia. Using fallback list.")
    tickers = pd.Series(["AAPL", "MSFT", "GOOGL", "AMZN", "META"])  # fallback

# ===============================================
# FUNCTION: Calculate MACD
# ===============================================
def calc_macd(data, len1=12, len2=26, len3=9):
    shortEMA = data.ewm(span=len1, adjust=False).mean()
    longEMA = data.ewm(span=len2, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=len3, adjust=False).mean()
    return MACD, signal

# ===============================================
# FUNCTION: Simple Trading Simulation
# ===============================================
def test_it(opens, closes, preds, start_account=1000, thresh=0):
    account = start_account
    changes = []

    for i in range(len(preds)):
        # If predicted price increase > threshold, simulate a "buy"
        if (preds[i] - opens[i]) / opens[i] >= thresh:
            account += account * (closes[i] - opens[i]) / opens[i]
        changes.append(account)

    changes = np.array(changes)
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(changes)), changes, label='Algorithmic Trading', color='green')
    plt.title("Trading Simulation Performance")
    plt.legend()
    plt.show()

    invest_total = start_account + start_account * (closes[-1] - opens[0]) / opens[0]
    print('📈 Simple Hold Total:', round(invest_total, 2),
          str(round((invest_total - start_account) / start_account * 100, 1)) + '%')
    print('🤖 Algo-Trading Total:', round(account, 2),
          str(round((account - start_account) / start_account * 100, 1)) + '%')

# ===============================================
# MAIN EXECUTION LOOP
# ===============================================
for ticker in tickers[:1]:   # Limit to 1 ticker for demo
    print(f"\n========== Analyzing {ticker} ==========")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=15 * 365)

    # Download 15 years of daily price data
    history = yf.download(ticker, start=start_date, end=end_date, interval='1d', prepost=False)
    if history.empty:
        print(f"⚠️ No data found for {ticker}, skipping...")
        continue

    history = history.loc[:, ['Open', 'Close', 'Volume']]

    # Feature engineering
    history['Prev_Close'] = history['Close'].shift(1)
    history['Prev_Volume'] = history['Volume'].shift(1)
    history['weekday'] = history.index.weekday

    # Simple Moving Averages
    for n in [5, 10, 20, 50, 100, 200]:
        history[f'{n}SMA'] = history['Prev_Close'].rolling(n).mean()

    # MACD indicators
    MACD, signal = calc_macd(history['Prev_Close'])
    history['MACD'] = MACD
    history['MACD_signal'] = signal

    # Drop missing / infinite values
    history = history.replace(np.inf, np.nan).dropna()

    # Train/test split
    y = history['Close']
    X = history.drop(['Close', 'Volume'], axis=1).values
    num_test = 365  # last 1 year for testing
    X_train, y_train = X[:-num_test], y[:-num_test]
    X_test, y_test = X[-num_test:], y[-num_test:]

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Plot predictions vs actual
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(y_test)), y_test, label='Actual', color='blue')
    plt.plot(range(len(preds)), preds, label='Predicted', color='red')
    plt.title(f"{ticker} - Actual vs Predicted Closing Prices")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    # Evaluate & simulate simple trading
    test_it(X_test.T[0], y_test.values, preds, 1000, 0)

print("\n✅ Stock Price Prediction Completed Successfully!")
