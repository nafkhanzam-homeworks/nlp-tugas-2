from typing import Iterable, Iterator, List, Tuple
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

import pandas as pd
import gensim

from cache_decorator import build_cache_decorator
from nergrit_framework import NergritFramework

cache = build_cache_decorator('nergrit-non-contextual-embeddings')


class NergritNonContextualEmbeddings(NergritFramework):
    @cache('word2vec_model')
    def build_word2vec_model(self):
        words = self.get_words()
        model = gensim.models.Word2Vec(
            sentences=[words],
            min_count=1,
            window=1,
            workers=8,
        )
        return model

    @cache('word2vec_train')
    def build_Xy_train(self, model):
        X, y = self._build_Xy(model, self.train_df,
                              self.build_tags(self.train_df))
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        return X, y

    @cache('word2vec_validation')
    def build_Xy_validation(self, model):
        return self._build_Xy(model, self.validation_df, self.build_tags(self.validation_df))

    @cache('word2vec_test')
    def build_X_test(self, model):
        X = []
        for word in self.test_series:
            X.append(model.wv[word])
        return X

    def build_label_encoder(self):
        le = preprocessing.LabelEncoder()
        le.fit(list(set(self.build_tags(self.train_df))))
        return le

    def _create_train_sentence_iter(self) -> Iterable[str]:
        df_list = self.readers.train_dataset_reader.read_to_sentence_dataframe_list()
        for df in df_list:
            yield df['token'].astype(str).to_list()

    def _build_Xy(self, model, df: pd.DataFrame, tags: List[str]):
        X_train = []
        for word in df['token']:
            X_train.append(model.wv[word])
        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(tags)

        return X_train, y_train

    def build_tags(self, df) -> List[str]:
        return df['label'].astype(str).to_list()

    @cache('vocab_words')
    def get_words(self):
        return self.uncased_vocab_series.astype(str).to_list() \
            + self.train_df['token'].astype(str).to_list()

    def build_X_sentences_test_iter(self, model) -> Iterator[pd.DataFrame]:
        for sentence_series in self.test_sentence_series:
            X = []
            for word in sentence_series:
                X.append(model.wv[word])
            yield X
