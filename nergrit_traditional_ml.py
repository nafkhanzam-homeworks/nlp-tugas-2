from typing import List

import pandas as pd

from nergrit_framework import NergritFramework


class NergritTraditionalML(NergritFramework):
    def __init__(self):
        super().__init__()
        self.train_df = self.readers.read_train_dataframe()
        self.validation_df = self.readers.read_validation_dataframe()
        self.test_series = self.readers.read_test_series()
        self.test_sentence_series = self.readers.read_test_sentence_series()
        self.uncased_vocab_series = self.readers.read_vocab_uncased_series()

    def build_naive_bayes_model(self):
        pass

    def _build_one_hot_encoded_dataframe(self, origin_df: pd.DataFrame) -> pd.DataFrame:
        if origin_df.columns['token'] is None:
            raise Exception('DataFrame does not have token column!')

        self.uncased_vocab_series
