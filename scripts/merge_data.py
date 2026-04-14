import pandas as pd
import yfinance as yf
import numpy as np
import os
import requests
import warnings
import warnings

warnings.filterwarnings('ignore')
print("--- ENFORCING HISTORICAL DATA SYNC (2021) ---")


# 1. MANUALLY SET DATES (Επέκταση δείγματος σε 1 χρόνο)
start_date = "2021-01-01"
end_date = "2022-01-01"

print(f"Targeting Period: {start_date} to {end_date}")

# 2. DOWNLOAD BTC
print("Downloading Bitcoin prices for Sept 2021...")
btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)
btc.reset_index(inplace=True)
btc.columns = [c.lower() for c in btc.columns]

# 2. DOWNLOAD CRYPTO DATA (BTC & ETH)
cryptos = ['BTC-USD', 'ETH-USD']
crypto_dfs = {}
for symbol in cryptos:
    print(f"Downloading {symbol} prices for {start_date} to {end_date}...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    crypto_dfs[symbol] = df

btc['date'] = pd.to_datetime(btc['date']).dt.tz_localize(None)

# 3. FETCH FEAR & GREED
print("Fetching Fear & Greed Index...")
fng_url = "https://api.alternative.me/fng/?limit=3000"
response = requests.get(fng_url).json()
fng_data = pd.DataFrame(response['data'])
fng_data['date'] = pd.to_datetime(fng_data['timestamp'].astype(float), unit='s').dt.tz_localize(None)
fng_data['fear_greed'] = fng_data['value'].astype(float)
fng_data = fng_data[['date', 'fear_greed']]

# 4. MERGE
print("Merging BTC and Sentiment...")
merged = pd.merge(btc, fng_data, on='date', how='left')
merged['fear_greed'] = merged['fear_greed'].ffill()

# 5. ECONOMETRICS
merged['log_returns'] = np.log(merged['close'] / merged['close'].shift(1))
# Reduced volatility window to 3 days because 7 is too much for a 16-day sample
merged['volatility'] = merged['log_returns'].rolling(window=3).std()

# 6. SAVE
out_path = os.path.join('exports', 'Master_Data.xlsx')

# 4. MERGE & SAVE FOR EACH CRYPTO
for symbol, df in crypto_dfs.items():
    print(f"Merging {symbol} and Sentiment...")
    merged = pd.merge(df, fng_data, on='date', how='left')
    merged['fear_greed'] = merged['fear_greed'].ffill()
    # 5. ECONOMETRICS & TECHNICAL INDICATORS
    merged['log_returns'] = np.log(merged['close'] / merged['close'].shift(1))
    merged['volatility'] = merged['log_returns'].rolling(window=5).std()
    # Rolling averages
    merged['close_rolling_mean_7'] = merged['close'].rolling(window=7).mean()
    merged['close_rolling_std_7'] = merged['close'].rolling(window=7).std()
    # Sentiment volatility (rolling std of fear_greed)
    merged['fear_greed_volatility_7'] = merged['fear_greed'].rolling(window=7).std()
    # Bull/Bear market dummy (1=close above 200-day SMA, 0=otherwise)
    merged['SMA_200'] = merged['close'].rolling(window=200).mean()
    merged['bull_market'] = (merged['close'] > merged['SMA_200']).astype(int)
    # RSI (14)
    delta = merged['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    merged['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = merged['close'].ewm(span=12, adjust=False).mean()
    ema26 = merged['close'].ewm(span=26, adjust=False).mean()
    merged['MACD'] = ema12 - ema26
    merged['MACD_signal'] = merged['MACD'].ewm(span=9, adjust=False).mean()
    # EMA & SMA
    merged['EMA_20'] = merged['close'].ewm(span=20, adjust=False).mean()
    merged['SMA_20'] = merged['close'].rolling(window=20).mean()
    # 6. SAVE
    out_path = os.path.join('exports', f'Master_Data_{symbol.replace("-USD","")}.xlsx')
    merged.dropna().to_excel(out_path, index=False)
    print(f"SUCCESS! Master Data for {symbol} created with {len(merged.dropna())} rows.")
merged.dropna().to_excel(out_path, index=False)
print(f"SUCCESS! Master Data created with {len(merged.dropna())} rows.")