from typing import List
from ..abstract_entity_engine import AbstractEntityEngine
import re


class PyAbsaEntityEngine(AbstractEntityEngine):
    def __init__(self, coins: List[str], **kwargs):
        """

        :param coins: a list of the full name of the cryptocurrencies to be analysed.
        """
        self.__dict__.update(kwargs)
        self.coins = coins

    def get_aspects(self, text: str) -> str:
        """
        This entity engine is specifically made for the PyABSA library and the format of input it accepts - [ASP] tokens.
        Unlike the other entity engine, this one returns the text with the [ASP] tokens placed around the aspects.
        !TODO optmize this using spacy's entity recognition training method (train NER)
        !TODO test this
        :param text: input
        :return: the text with the [ASP] tokens *correctly* placed around each aspect.
        """
        for coin in self.coins:
            if coin in text:
                position_inc = 0  # Deals with changing length of original array.
                matches = re.finditer(rf'\b{coin}\b', text)
                matches = list(matches)
                multiple = len(matches) > 1
                last = len(matches) - 1

                for current, match in enumerate(matches):
                    s, e = match.start() + position_inc, match.start() + len(coin) + position_inc
                    ASP_start_indices = [m.start() for m in re.finditer('ASP]', text)]  # This prevents [ASP] tokens placement around bitcoin in the following (and similar) case: [ASP]Bitcoin cash[ASP] is nice!
                    if not (s - 4 in [si for si in ASP_start_indices]): # -4, because we conside 4 tokens. we don't consider [ASP], because it is interpreted as reuglar expression
                        text = text[:s] + '[ASP]' + text[s:e] + '[ASP]' + text[e:]
                        if multiple and not (current == last):
                            position_inc += 10
        return text
