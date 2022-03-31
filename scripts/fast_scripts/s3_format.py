import pandas as pd

df = pd.read_csv('../../data/res_preprocessed.csv')
def put_quotes(text):
    if text.startswith('""') and text.endswith('"'):
        print('Text is: ', text)
        return text
    else:
        return f'"{text}"'

print(df)
# df['embedded_text'] = df['embedded_text'].apply(put_quotes)
df = df[['embedded_text']]
df.to_csv('embedded_text.csv', index=False)
print(df)
