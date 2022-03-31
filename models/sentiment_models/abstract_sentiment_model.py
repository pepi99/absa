from abc import ABC, abstractmethod


class AbstractSentimentModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_sentiments(self):
        pass
