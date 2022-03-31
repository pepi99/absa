from preprocessors.abstract_preprocessors.abstract_preprocessor import Preprocessor
import re


class PyAbsaTwitterPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def preprocess(self, text):
        """

        :param text: input
        :return: standard twitter preprocessing
        """
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('\n', ' ') # remove nl symbols
        preprocessed = re.sub(r'https?:\/\/\S+', '', preprocessed) # https links removal
        preprocessed = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', preprocessed) # wwww / .com strings removal.

        for coin_key in self.coins_dict.keys(): # Mapping: eth -> ethereum.
            coin_val = self.coins_dict[coin_key]
            preprocessed = re.sub(rf'\b{coin_key}\b', f'{coin_val}', preprocessed)

        return preprocessed
