from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
from cached import cached

from nergrit_framework import NergritFramework


class NergritTraditionalML(NergritFramework):
    def __init__(self):
        super().__init__()
        self.train_df = self._build_train_df()
        self.validation_df = self._build_validation_df()
        self.test_series = self._build_test_series()
        self.test_sentence_series = self._build_test_sentence_series()
        self.uncased_vocab_series = self._build_uncased_vocab_series()

    def build_naive_bayes_model(self) -> Tuple[pd.DataFrame, pd.Series]:
        preprocessed_train_df = \
            self._build_one_hot_encoded_dataframe(self.train_df)
        return preprocessed_train_df

    @cached('nergrit_train_df.pkl')
    def _build_train_df(self):
        return self.readers.read_train_dataframe()

    @cached('nergrit_validation_df.pkl')
    def _build_validation_df(self):
        return self.readers.read_validation_dataframe()

    @cached('nergrit_test_series.pkl')
    def _build_test_series(self):
        return self.readers.read_test_series()

    @cached('nergrit_test_sentence_series.pkl')
    def _build_test_sentence_series(self):
        return self.readers.read_test_sentence_series()

    @cached('nergrit_uncased_vocab_series.pkl')
    def _build_uncased_vocab_series(self):
        return self.readers.read_vocab_uncased_series()

    def _build_one_hot_encoded_dataframe(self, origin_df: pd.DataFrame) -> pd.DataFrame:
        if not 'token' in origin_df.columns:
            raise Exception('DataFrame does not have token column!')

        df = origin_df.copy()

        tf_vectorizer = CountVectorizer()
        tf_vectorizer.fit(self.uncased_vocab_series)

        X_train = pd.DataFrame(
            tf_vectorizer.transform(df['token']).toarray(),
            columns=tf_vectorizer.get_feature_names(),
        )
        y_train = df['label']

        return X_train, y_train
