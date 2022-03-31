import pandas as pd
from preprocessors.preprocessors.pyabsa_preprocessor import PyAbsaTwitterPreprocessor
import yaml

def main():
    with open('../../config/coins_config.yaml') as stream:
        CONFIG = yaml.safe_load(stream)

    coins_dict = CONFIG['coins-dict']
    coins = coins_dict.values()

    config = {'engine_config': {'coins': coins},
              'preprocessor_config': {'coins_dict': coins_dict},
              'senti_config': {'build_pipeline': False}}

    twitter_preprocessor = PyAbsaTwitterPreprocessor(coins_dict=coins_dict)

    with open('../../data/crypto_train.txt', 'r') as f:
        data = f.readlines()
        print(data)
    data = [x[:-1] for x in data]
    texts = data[0::3]
    aspects = data[1::3]
    labels = data[2::3]
    texts = [twitter_preprocessor.preprocess(text) for text in texts]
    a = 1
    pass

main()
