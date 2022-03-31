import pandas as pd
from models.entity_engines.entity_engines.pyabsa_entity_engine import PyAbsaEntityEngine
from models.sentiment_models.pyabsa_sentiment import PyabsaSentimentAnalyser
from preprocessors.preprocessors.pyabsa_preprocessor import PyAbsaTwitterPreprocessor
from models.base_model import BaseModel
import yaml
from tqdm import tqdm

tqdm.pandas()


def give_sentiment_labels(texts, model, coins):
    sents = []
    scores = []

    for text in tqdm(texts):
        sentiments = model.sentiment(text)
        res = [sentiments[k]['sentiment'] for k in coins]
        scrs = [sentiments[k]['confidence'] for k in coins]
        sents.append(res)
        scores.append(scrs)
    return sents, scores


def main():
    df = pd.read_csv('../../data/tweets.csv')

    df['embedded_text'] = df['embedded_text'].apply(lambda x: str(x).lower())
    coins = ['bitcoin', 'ethereum', 'ultracoin', 'litecoin', 'ripple', 'stellar', 'bitcoin cash', 'binance coin',
             'tron', 'chainlink']

    df = df[df['embedded_text'].str.contains('|'.join(coins))]
    df = df[['embedded_text']]
    df = df.sample(frac=1)
    df = df[:35]
    texts = df['embedded_text'].values

    with open('../../config/coins_config.yaml') as stream:
        CONFIG = yaml.safe_load(stream)

    coins_dict = CONFIG['coins-dict']
    coins = coins_dict.values()

    config = {'engine_config': {'coins': coins},
              'preprocessor_config': {'coins_dict': coins_dict},
              'senti_config': {'build_pipeline': False,
                               'coins': coins,
                               'checkpoint': '/Users/petar.ulev/Documents/absa/checkpoints/pyabsa_checkpoints/fast_lsa_t_Crypto_acc_83.62_f1_84.14_state_dict'
                               }
              }

    base_model = BaseModel(PyAbsaEntityEngine, PyAbsaTwitterPreprocessor, PyabsaSentimentAnalyser,
                           engine_config=config['engine_config'],
                           preprocessor_config=config['preprocessor_config'],
                           senti_config=config['senti_config']
                           )

    sents, scores = give_sentiment_labels(texts, base_model, coins)

    df['sentiment'] = sents
    df['confidence'] = scores
    df['coins'] = [coins] * (df.shape[0])

    df = df.explode(['sentiment', 'confidence', 'coins'])
    df = df.dropna()

    preprocessor = PyAbsaTwitterPreprocessor(coins_dict=coins_dict)
    df['embedded_text'] = df['embedded_text'].apply(preprocessor.preprocess)
    df = df[['embedded_text', 'sentiment', 'confidence', 'coins']]
    # df.reset_index(inplace=True)
    df.to_csv('../../data/res_preprocessed.csv', index=False)


main()
