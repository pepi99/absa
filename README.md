# General

The code creates a pipeline with custom components and configuration for aspect-based sentiment analysis. 
Currently, one model has been developed (with all its sub-components, please see details here: https://arxiv.org/abs/2110.08604).
The implementation can be found in the open-source library https://github.com/yangheng95/PyABSA (also created by the author of the paper).
This library essentially wraps PyABSA, includes some pre-trained models (using PyABSA) on custom labelled crypto texts.



# Structure

The pipeline consists of 3 components: 
1. A trained model component
2. Preprocessing component
3. Entity recognition engine component

Because our task is limited to fixed entities, (can be seen in the config folder), we are not using a pre-trained NER
(neither training it ourselves), because the problem simplifies to just using string manipulations. 

# Note!

The pipeline `IS NOT` used for training. It just wraps a couple of components to make easier structuring the code. If you want 
to train a model yourself, you can just create a new sentiment model component and integrate it in the code (see PyABSA or scalac's implementation).
Put your model in one of the checkpoints directories

# How it works

Before running the pipeline, you have to load the coins config folder. It contains all the coins we are going to analyse and 
also their mappings from full to small names. An example script:

```from models.entity_engines.entity_engine_components.pyabsa_entity_engine import PyAbsaEntityEngine
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
    ```