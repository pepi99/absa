import pandas as pd
import yaml
from preprocessors.preprocessors.absa_scalac_twitter_preprocessor import TwitterPreprocessor
import re

with open('../../config/coins_config.yaml') as stream:
    CONFIG = yaml.safe_load(stream)

coins_dict = CONFIG['coins-dict']
coins = coins_dict.values()

config = {'engine_config': {'coins': coins},
          'preprocessor_config': {'coins_dict': coins_dict},
          'senti_config': {'build_pipeline': False}}

twitter_preprocessor = TwitterPreprocessor(coins_dict=coins_dict)

print(coins)
def get_aspects(text):
    return [coin for coin in coins if coin in text]

def get_positions(text, aspect):

    return [(m.start(), m.start() + len(aspect)) for m in re.finditer(aspect, text)]

# print(get_positions('All the appetizers and salads were fabulous, the steak was mouth watering and the pasta was delicious!!!', ''))
df = pd.read_csv('../../data/ico_data.csv')
df['text'] = df['text'].apply(twitter_preprocessor.preprocess)
df['aspects'] = df['text'].apply(get_aspects)


df = df.explode('aspects')
df.dropna(inplace=True)

df['aspect_position'] = df.apply(lambda x: get_positions(x['text'], x['aspects']), axis=1)
df = df.explode('aspect_position')
df = df[df['aspects'] == 'bitcoin']

print(df[1:10])
print(df.shape)
df.to_csv('aaaa.csv', index=False)

