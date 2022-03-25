from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB

from cache_decorator import build_cache_decorator
from nergrit_framework import NergritFramework

cache = build_cache_decorator('nergrit')


class NergritTraditionalML(NergritFramework):
    def init_data(self):
        self.train_df = self._build_train_df()
        self.validation_df = self._build_validation_df()
        self.test_series = self._build_test_series()
        self.test_sentence_series = self._build_test_sentence_series()
        self.uncased_vocab_series = self._build_uncased_vocab_series()

    @cache('nb_model')
    def build_naive_bayes_model(self) -> GaussianNB:
        X_train = self.build_X_train()
        y_train = self.get_y_train()

        model = GaussianNB()
        model.fit(X_train, y_train)

        return model

    @cache('nb_X_train')
    def build_X_train(self) -> pd.DataFrame:
        return self._build_one_hot_encoded_dataset(
            self.train_df
        )

    def get_y_train(self):
        return self.train_df['label']

    @cache('nb_X_validation')
    def build_X_validation(self) -> pd.DataFrame:
        return self._build_one_hot_encoded_dataset(
            self.validation_df
        )

    def get_y_validation(self):
        return self.validation_df['label']

    @cache('nb_X_test')
    def build_X_test(self) -> pd.DataFrame:
        return self._build_one_hot_encoded_dataset(
            self.test_series.to_frame('token')
        )

    @cache('train_df')
    def _build_train_df(self):
        return self.readers.read_train_dataframe()

    @cache('validation_df')
    def _build_validation_df(self):
        return self.readers.read_validation_dataframe()

    @cache('test_series')
    def _build_test_series(self):
        return self.readers.read_test_series()

    @cache('test_sentence_series')
    def _build_test_sentence_series(self):
        return self.readers.read_test_sentence_series()

    @cache('uncased_vocab_series')
    def _build_uncased_vocab_series(self):
        return self.readers.read_vocab_uncased_series()

    def _build_one_hot_encoded_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if not 'token' in df.columns:
            raise Exception('DataFrame does not have token column!')

        tf_vectorizer = self._build_tf_vectorizer()

        token_list = df['token'].apply(lambda x: x.lower())

        X_train = pd.DataFrame(
            tf_vectorizer.transform(token_list).toarray(),
            columns=tf_vectorizer.get_feature_names(),
        )

        return X_train

    @cache('tf_vectorizer')
    def _build_tf_vectorizer(self):
        tf_vectorizer = CountVectorizer()
        tf_vectorizer.fit(self.uncased_vocab_series)

        return tf_vectorizer
