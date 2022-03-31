class Pipeline:
    def __init__(self, EntityEngine, Preprocessor, SentiAnalyser, config={}):
        """

        :param EntityEngine:
        :param Preprocessor:
        """
        self._entity_engine = EntityEngine(**config['engine_config'])
        self._preprocessor = Preprocessor(**config['preprocessor_config'])
        self._sentiment_analyser = SentiAnalyser(**config['senti_config'])

    def sentiment(self, text: str) -> dict:
        """

        :param text:
        :return: sentiment dictionary
        """
        text = self._preprocessor.preprocess(text)
        aspects = self._entity_engine.get_aspects(text)
        sentiments = self._sentiment_analyser.get_sentiments(text, aspects)

        return sentiments
