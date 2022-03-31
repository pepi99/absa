from preprocessors.abstract_preprocessors.abstract_preprocessor import Preprocessor
import re


class TwitterPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def preprocess(self, text):
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('\n', ' ')
        preprocessed = re.sub(r'https?:\/\/\S+', '', preprocessed)
        preprocessed = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', preprocessed)

        for coin in self.coins_dict.keys():
            if coin in preprocessed:
                preprocessed = preprocessed.replace(coin, self.coins_dict[coin])

        return preprocessed
