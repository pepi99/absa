import aspect_based_sentiment_analysis as absa
from typing import List


class ScalacSentimentAnalyser:
    def __init__(self, **kwargs):
        """

        :param kwargs: keyword arguments
        """

        self.__dict__.update(kwargs)

        if not self.build_pipeline:  # Use the default absa pipeline (which in our case is good)
            self.nlp = absa.load()

    def get_sentiments(self, text: str, aspects: List[str]) -> dict:
        """

        :param text: input
        :param aspects:
        :return: dictionary with sentiment data
        """
        try:
            scalac_sentiments = self.nlp(text, aspects=aspects)

        except Exception as e:
            scalac_sentiments = {}

        if scalac_sentiments:  # Could be empty
            scalac_sentiments = {example.aspect: {'sentiment_cat': self.__labeller(example.sentiment, numerical=True),
                                           'sentiment_str': self.__labeller(example.sentiment, numerical=False),
                                           'scores': example.scores}
                          for example in scalac_sentiments.examples}

        for coin in self.coins:
            if coin not in scalac_sentiments.keys():
                scalac_sentiments[coin] = {'sentiment_cat': None,
                                    'sentiment_str': None,
                                    'scores': None}

        return scalac_sentiments


    @staticmethod
    def __labeller(sent, numerical=True):
        if numerical:
            if sent == absa.Sentiment.neutral:
                return '0'
            if sent == absa.Sentiment.negative:
                return '1'
            if sent == absa.Sentiment.positive:
                return '2'
            else:
                return None
        else:
            if sent == absa.Sentiment.neutral:
                return 'neutral'
            if sent == absa.Sentiment.negative:
                return 'negative'
            if sent == absa.Sentiment.positive:
                return 'positive'
            else:
                return None
