from models.entity_engines.entity_engine_components.pyabsa_entity_engine import PyAbsaEntityEngine
from models.sentiment_models.sentiment_model_components.pyabsa_sentiment import PyabsaSentimentAnalyser
from preprocessors.preprocessors.pyabsa_preprocessor import PyAbsaTwitterPreprocessor
from models.pipeline import Pipeline
import yaml


with open('../../config/coins_config.yaml') as stream:
    CONFIG = yaml.safe_load(stream)


coins_dict = CONFIG['coins-dict']
coins = coins_dict.values()

config = {'engine_config': {'coins': coins},
          'preprocessor_config': {'coins_dict': coins_dict},
          'senti_config': {'build_pipeline': False,
                           'coins': coins,
                           'checkpoint': '/Users/petar.ulev/Documents/absa/checkpoints/pyabsa_checkpoints/newest_model/fast_lsa_t_Crypto_acc_86.41_f1_85.88'
                           }
          }


def main():
    """
    Run model
    :return:
    """
    base_model = Pipeline(PyAbsaEntityEngine, PyAbsaTwitterPreprocessor, PyabsaSentimentAnalyser, config)

    texts = ['Block CEO, Robinhood CEO comments on Twitter Over the high Bitcoin Fees',
             'What do you think about the recent news concerning btc?',
             'I think btc and eth about to dump. And cardano about to pump.',
             'How do u see the crypto world going? #btc']
    for text in texts:
        sentiments = base_model.sentiment(text)

        print(f'Text is : |*{text}*| and deberta-v3-large sentiment is: {sentiments}')


if __name__ == '__main__':
    main()
