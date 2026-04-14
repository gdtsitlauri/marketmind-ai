import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

print("--- STARTING KAGGLE REDDIT AI SENTIMENT ANALYSIS ---")

# 1. LOAD KAGGLE DATA
file_path = os.path.join('data', 'reddit_cc.csv')

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the CSV is in the 'data' folder and named 'reddit_cc.csv'.")
    exit()

print("Loading historical dataset...")
# Read the CSV. The common Kaggle crypto reddit datasets use these columns:
try:
    df = pd.read_csv(file_path, usecols=['title', 'body', 'timestamp'])
    df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')
    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
except ValueError:
    print("Warning: Standard column names not found. Attempting generic read...")
    df = pd.read_csv(file_path)
    # Fallback: assume first col is title, second is body
    df['text'] = df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str)
    # Try to find a date column
    date_col = 'timestamp' if 'timestamp' in df.columns else df.columns[2]
    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date

# Clean empty rows and drop any that failed date parsing
df = df[df['text'].str.strip() != '']
df = df.dropna(subset=['date'])
print(f"Successfully loaded {len(df)} historical posts.")

# Limit to the most recent 10,000 entries to save processing time
if len(df) > 10000:
    print("Dataset is very large. Processing the most recent 10,000 entries...")
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False)
    df = df.head(10000)


# 2. AI SETUP (FINBERT & RoBERTa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading AI Models (FinBERT & RoBERTa) on: {device}")

# FinBERT
finbert_name = "ProsusAI/finbert"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_name).to(device)

# RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)
from transformers import pipeline
roberta_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0 if torch.cuda.is_available() else -1)


# 3. SENTIMENT ANALYSIS (FinBERT & RoBERTa)
finbert_results = []
roberta_results = []
print("AI is analyzing historical text sentiment with FinBERT & RoBERTa. This may take a few minutes...")

finbert_model.eval()
with torch.no_grad():
    for text in tqdm(df['text']):
        # FinBERT
        inputs = finbert_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        finbert_results.append(probs)
        # RoBERTa
        roberta_pred = roberta_pipe(text[:512])[0]
        roberta_results.append(roberta_pred['label'])

# FinBERT output order: [Positive, Negative, Neutral]
df[['positive', 'negative', 'neutral']] = finbert_results
df['roberta_sentiment'] = roberta_results

# 3b. TOPIC MODELING (LDA)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
print("Running topic modeling (LDA)...")
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda.fit_transform(X)
df['topic'] = lda_topics.argmax(axis=1)

# 4. GROUP BY DATE
print("Aggregating daily sentiment scores...")
final_report = df.groupby('date').agg({
    'positive': 'mean',
    'negative': 'mean',
    'neutral': 'mean',
    'text': 'count'
}).rename(columns={'text': 'post_count'}).reset_index()

# 5. SAVE EXPORT
if not os.path.exists('exports'):
    os.makedirs('exports')

file_name = os.path.join('exports', 'Reddit_AI_Sentiment.xlsx')
final_report.to_excel(file_name, index=False)

print(f"\n--- SUCCESS! ---")
print(f"Historical AI processing complete. Data saved to '{file_name}'.")
print("You can now run 'final_merge.py' to update the Ultimate Data file!")