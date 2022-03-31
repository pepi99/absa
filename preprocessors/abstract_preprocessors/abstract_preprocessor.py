import abc


class Preprocessor(abc.ABC):

    @abc.abstractmethod
    def preprocess(self, text):
        pass



