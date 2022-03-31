from typing import List
from abc import ABC, abstractmethod


class AbstractEntityEngine(ABC):
    def __init__(self, coins: List[str], **kwargs):
        """

        :param coins: a list of the full name of the cryptocurrencies to be analysed.
        """
        self.__dict__.update(kwargs)
        self.coins = coins

    @abstractmethod
    def get_aspects(self):
        """
        Get aspects (could be implemented in different ways)
        """
        pass


