from typing import List
class EntityRecogniser:
    def __init__(self, coins: List[str], **kwargs):
        """

        :param coins: a list of the full name of the cryptocurrencies to be analysed.
        """
        self.__dict__.update(kwargs)
        self.coins = coins

    def get_aspects(self, text: str) -> List[str]:
        """

        :param text: input
        :return: only those coins that are present in the text
        """
        aspects = [coin for coin in self.coins if coin in text]
        return aspects


