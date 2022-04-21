import pandas as pd
from models.entity_engines.entity_engine_components.pyabsa_entity_engine import PyAbsaEntityEngine
from models.sentiment_models.sentiment_model_components.pyabsa_sentiment import PyabsaSentimentAnalyser
from preprocessors.preprocessors.pyabsa_preprocessor import PyAbsaTwitterPreprocessor
from models.pipeline import Pipeline
import yaml
from tqdm import tqdm
from scripts.fast_scripts.compare import create_comp_df

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
    return texts, sents, scores


def main():
    df = pd.read_csv('../../data/tweets_ulev_baby.csv')
    df['embedded_text'] = df['embedded_text'].apply(lambda x: str(x).lower())
    coins = ['bitcoin', 'ethereum', 'ultracoin', 'litecoin', 'ripple', 'stellar', 'bitcoin cash', 'binance coin',
             'tron', 'chainlink']

    df = df[df['embedded_text'].str.contains('|'.join(coins))]
    df = df[['embedded_text']]
    # df = df.sample(frac=1)
    df = df[23000:23100]
    # df = df[:100]
    print('Data is: ', df)
    texts = df['embedded_text'].values

    with open('../../config/coins_config.yaml') as stream:
        CONFIG = yaml.safe_load(stream)

    coins_dict = CONFIG['coins-dict']
    coins = coins_dict.values()
    preprocessor = PyAbsaTwitterPreprocessor(coins_dict=coins_dict)

    model_checkpoints = ['/Users/petar.ulev/Documents/absa/checkpoints/pyabsa_checkpoints/newest_model/fast_lsa_t_Crypto_acc_86.41_f1_85.88',
                         '/Users/petar.ulev/Documents/absa/checkpoints/pyabsa_checkpoints/fast_lsa_t_Crypto_acc_83.62_f1_84.14_state_dict']
    dfs = []
    for idx, model_checkpoint in enumerate(model_checkpoints):
        config = {'engine_config': {'coins': coins},
                  'preprocessor_config': {'coins_dict': coins_dict},
                  'senti_config': {'build_pipeline': False,
                                   'coins': coins,
                                   'checkpoint': model_checkpoint
                                   }
                  }

        base_model = Pipeline(PyAbsaEntityEngine, PyAbsaTwitterPreprocessor, PyabsaSentimentAnalyser, config)

        texts, sents, scores = give_sentiment_labels(texts, base_model, coins)
        df_new = pd.DataFrame({})
        df_new['sentiment'] = sents
        df_new['confidence'] = scores
        df_new['coin'] = [coins] * (df_new.shape[0])
        df_new['embedded_text'] = [preprocessor.preprocess(text) for text in texts]

        df_new = df_new.explode(['sentiment', 'confidence', 'coin'])
        df_new = df_new.dropna()

        dfs.append(df_new)
    #dfs[0].to_csv('data/tweets_ulev_baby_labelled.csv', index=False)
    create_comp_df(dfs)


main()
