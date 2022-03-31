from pyabsa import APCCheckpointManager
import itertools


class PyabsaSentimentAnalyser:
    """
    Wrapper of the pyabsa implementation of algorithms.
    """

    def __init__(self, checkpoint, **kwargs):
        self.__dict__.update(kwargs)
        self.model = APCCheckpointManager.get_sentiment_classifier(checkpoint=checkpoint)

    def get_sentiments(self, text: str, aspects: str) -> dict:
        """
        If an aspect appears a couple of times, take only the last occurrence (final decision).
        Think about other options as well.
        Example output: {'ethereum': {'sentiment': 'Negative', 'confidence': 0.3634795546531677}, 'bitcoin': {'sentiment': 'Negative', 'confidence': 0.3637586534023285}}
        :param text: input
        :param aspects: this is actually the text with the [ASP] tokens placed around each coin. This is a design choice to adhere to the present architecture in the base_pipeline. So we can basically ignore the text here.
        :return: dictionary of sentiments
        """
        # Check if aspects were found:
        ct = aspects.count('[ASP]')
        if (ct % 2 == 0) and ct > 0:  # [ASP] token appears even number of times, also more than 0
            pyabsa_sents = self.model.infer(aspects, print_result=False)
            pyabsa_sents = {
                'sentiment': list(itertools.chain(*[pyabsa_sents[j]['sentiment'] for j in range(len(pyabsa_sents))])),
                'confidence': list(itertools.chain(*[pyabsa_sents[j]['confidence'] for j in range(len(pyabsa_sents))])),
                'aspect': list(itertools.chain(*[pyabsa_sents[j]['aspect'] for j in range(len(pyabsa_sents))]))
            }

            sentiment_dict = {coin: {'sent_conf': [(pyabsa_sents['sentiment'][i], pyabsa_sents['confidence'][i])
                                                   for i in range(len(pyabsa_sents['aspect']))
                                                   if pyabsa_sents['aspect'][i] == coin][-1]}
                              for coin in set(pyabsa_sents['aspect'])}

            new_sentiment_dict = {coin: {'sentiment': sentiment_dict[coin]['sent_conf'][0],
                                         'confidence': sentiment_dict[coin]['sent_conf'][1]}
                                  for coin in sentiment_dict.keys()}
        else:
            new_sentiment_dict = {}

        for coin in self.coins: # Put none if the coin is not found
            if coin not in new_sentiment_dict.keys():
                new_sentiment_dict[coin] = {'sentiment': None, 'confidence': None}

        return new_sentiment_dict
