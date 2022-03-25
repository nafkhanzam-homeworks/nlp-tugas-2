from typing import Iterator, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB

from cache_decorator import build_cache_decorator
from nergrit_framework import NergritFramework

cache = build_cache_decorator('nergrit')


class NergritTraditionalML(NergritFramework):
    _tf_vectorizer = None

    @cache('nb_model')
    def build_naive_bayes_model(self) -> GaussianNB:
        X_train = self.build_X_train()
        y_train = self.get_y_train()

        model = GaussianNB()
        model.fit(X_train, y_train)

        return model

    def test_naive_bayes_model(self, tokens: List[str]) -> List[str]:
        model = self.build_naive_bayes_model()
        return model.predict(
            self._build_one_hot_encoded_dataset(
                pd.DataFrame(tokens, columns=['token'])
            )
        )

    @cache('nb_X_train')
    def build_X_train(self) -> pd.DataFrame:
        X_train = self._build_one_hot_encoded_dataset(
            self.train_df
        )
        X_train.loc[len(X_train)] = [0] * len(X_train.columns)
        return X_train

    def get_y_train(self):
        y_train = self.train_df['label']
        y_train = y_train.append(pd.Series('O'), ignore_index=True)
        return y_train

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

    def build_X_sentences_test_iter(self) -> Iterator[pd.DataFrame]:
        for sentence_series in self.test_sentence_series:
            yield self._build_one_hot_encoded_dataset(
                sentence_series.to_frame('token')
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

    def _build_tf_vectorizer(self):
        if self._tf_vectorizer:
            return self._tf_vectorizer

        self._tf_vectorizer = CountVectorizer()
        self._tf_vectorizer.fit(self.train_df['token'])

        return self._tf_vectorizer
