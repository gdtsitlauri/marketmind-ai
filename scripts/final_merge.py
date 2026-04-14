import pandas as pd
import os

print("--- STARTING FINAL DATA MERGE ---")

# 1. Find all Master_Data_*.xlsx files
import glob
reddit_file = os.path.join('exports', 'Reddit_AI_Sentiment.xlsx')
master_files = glob.glob(os.path.join('exports', 'Master_Data_*.xlsx'))

# 2. Load and concatenate all crypto master files
print("Loading datasets...")
master_dfs = []
for f in master_files:
	df = pd.read_excel(f)
	# Add a column for the crypto symbol
	symbol = os.path.basename(f).replace('Master_Data_','').replace('.xlsx','')
	df['crypto'] = symbol
	master_dfs.append(df)
master_df = pd.concat(master_dfs, ignore_index=True)

reddit_df = pd.read_excel(reddit_file)

# 3. Format dates to ensure a perfect match during merge
master_df['date'] = pd.to_datetime(master_df['date']).dt.date
reddit_df['date'] = pd.to_datetime(reddit_df['date']).dt.date

# 4. Perform the final merge (Left join to keep all market days)
print("Merging market data with AI sentiment...")
final_df = pd.merge(master_df, reddit_df, on='date', how='left')

# 5. Export the final combined dataset
final_output_xlsx = os.path.join('exports', 'ULTIMATE_Data.xlsx')
final_output_csv = os.path.join('exports', 'ULTIMATE_Data.csv')
final_df.to_excel(final_output_xlsx, index=False)
final_df.to_csv(final_output_csv, index=False, encoding='utf-8')

# 6. Create data dictionary
data_dict = {
	'date': 'Ημερομηνία (YYYY-MM-DD)',
	'open': 'Τιμή ανοίγματος',
	'high': 'Υψηλότερη τιμή ημέρας',
	'low': 'Χαμηλότερη τιμή ημέρας',
	'close': 'Τιμή κλεισίματος',
	'adj close': 'Τιμή κλεισίματος (προσαρμοσμένη)',
	'volume': 'Όγκος συναλλαγών',
	'fear_greed': 'Δείκτης Fear & Greed',
	'log_returns': 'Λογάριθμος ημερήσιας απόδοσης',
	'volatility': 'Κυλιόμενη τυπική απόκλιση log_returns (5 μέρες)',
	'close_rolling_mean_7': 'Κυλιόμενος μέσος όρος τιμής (7 μέρες)',
	'close_rolling_std_7': 'Κυλιόμενη τυπική απόκλιση τιμής (7 μέρες)',
	'fear_greed_volatility_7': 'Κυλιόμενη τυπική απόκλιση Fear & Greed (7 μέρες)',
	'SMA_200': 'Simple Moving Average 200 ημερών',
	'bull_market': '1=Bull market (close > SMA_200), 0=Bear',
	'RSI_14': 'Relative Strength Index (14 μέρες)',
	'MACD': 'MACD indicator',
	'MACD_signal': 'MACD signal line',
	'EMA_20': 'Exponential Moving Average (20 μέρες)',
	'SMA_20': 'Simple Moving Average (20 μέρες)',
	'crypto': 'Crypto symbol (BTC, ETH, κλπ)',
	'positive': 'Μέσος ημερήσιος θετικός AI sentiment score',
	'negative': 'Μέσος ημερήσιος αρνητικός AI sentiment score',
	'neutral': 'Μέσος ημερήσιος ουδέτερος AI sentiment score',
	'roberta_sentiment': 'RoBERTa sentiment label',
	'topic': 'LDA topic label',
	'post_count': 'Αριθμός reddit posts ανά ημέρα',
}
dict_lines = [f"{col}: {desc}" for col, desc in data_dict.items() if col in final_df.columns]
with open(os.path.join('exports', 'data_dictionary.txt'), 'w', encoding='utf-8') as f:
	f.write("DATA DICTIONARY\n================\n")
	f.write("\n".join(dict_lines))

print(f"\n--- SUCCESS! ---")
print(f"The ultimate dataset has been created: {final_output_xlsx} and {final_output_csv}")
print("Data dictionary: exports/data_dictionary.txt")
print("The file is ready for Stata analysis!")