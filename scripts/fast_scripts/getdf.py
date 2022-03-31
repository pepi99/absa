import pandas as pd


df = pd.read_csv('../../data/tweets.csv')
df['embedded_text'] = df['embedded_text'].apply(lambda x: str(x).lower())
coins = ['bitcoin', 'ethereum', 'ultracoin', 'litecoin', 'ripple', 'stellar','bitcoin cash','binance coin','tron', 'chainlink']

df = df[df['embedded_text'].str.contains('|'.join(coins))]
df = df[['embedded_text']]
df.to_csv('../data/crypto_tweets.csv')