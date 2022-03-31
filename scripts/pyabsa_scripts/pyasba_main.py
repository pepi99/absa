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
                           'checkpoint': '/Users/petar.ulev/Documents/absa/checkpoints/pyabsa_checkpoints/fast_lsa_t_Crypto_acc_74.57_f1_75.27_state_dict'
                           }
          }


def main():
    """am
    Run model
    :return:
    """
    base_model = Pipeline(PyAbsaEntityEngine, PyAbsaTwitterPreprocessor, PyabsaSentimentAnalyser, config)

    text = "#cardano about to smash $1! what an insane run!   you can trade $ada along with other top crypto like #bitcoin and #ethereum on phemex.   get 10% off fees and up to a $750 trading bonus (limited time offer) when using this chainlink to sign up  18 17 175"
    sentiments = base_model.sentiment(text)

    print('Output: ', sentiments)


if __name__ == '__main__':
    main()
