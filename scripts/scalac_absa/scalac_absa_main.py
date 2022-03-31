from models.entity_engines.entity_engines.absa_scalac_entity_engine import EntityRecogniser
from models.sentiment_models.scalac_absa import ScalacSentimentAnalyser
from preprocessors.preprocessors.absa_scalac_twitter_preprocessor import TwitterPreprocessor
from models.base_model import BaseModel
import yaml


with open('../../config/coins_config.yaml') as stream:
    CONFIG = yaml.safe_load(stream)


coins_dict = CONFIG['coins-dict']
coins = coins_dict.values()

config = {'engine_config': {'coins': coins},
          'preprocessor_config': {'coins_dict': coins_dict},
          'senti_config': {'build_pipeline': False,
                           'coins': coins}}


def main():
    base_model = BaseModel(EntityRecogniser, TwitterPreprocessor, ScalacSentimentAnalyser,
                           engine_config=config['engine_config'],
                           preprocessor_config=config['preprocessor_config'],
                           senti_config=config['senti_config']
                           )

    text = "for the latest #cryptocurrency #bitcoin #blockchain trends, reviews, guides and memes follow  @boxminingnews  on twitter! 2 5 12"
    sentiments = base_model.sentiment(text)

    print('Output: ', sentiments)


if __name__ == '__main__':
    main()