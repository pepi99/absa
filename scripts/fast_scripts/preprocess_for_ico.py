import pandas as pd
from preprocessors.preprocessors.pyabsa_preprocessor import PyAbsaTwitterPreprocessor
import yaml
from tqdm import tqdm

tqdm.pandas()




def main():
    df = pd.read_csv('../../data/tweets.csv')

    df['embedded_text'] = df['embedded_text'].apply(lambda x: str(x).lower())
    coins = ['bitcoin', 'ethereum', 'ultracoin', 'litecoin', 'ripple', 'stellar', 'bitcoin cash', 'binance coin',
             'tron', 'chainlink']

    df = df[df['embedded_text'].str.contains('|'.join(coins))]
    df = df[['embedded_text']]
    df = df.sample(frac=1)
    with open('../../config/coins_config.yaml') as stream:
        CONFIG = yaml.safe_load(stream)

    coins_dict = CONFIG['coins-dict']
    preprocessor = PyAbsaTwitterPreprocessor(coins_dict=coins_dict)
    df['embedded_text'] = df['embedded_text'].apply(preprocessor.preprocess)

    df.to_csv('../../data/georgi_data.csv', index=False)


main()
