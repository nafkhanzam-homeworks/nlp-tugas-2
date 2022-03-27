from dataset_readers.complete_dataset_reader import CompleteDatasetReader

from cache_decorator import build_cache_decorator

cache = build_cache_decorator('nergrit')


class NergritFramework():
    def __init__(self):
        self.readers = CompleteDatasetReader(dataset_name="nergrit_ner-grit")

    def init_data(self, ignore_case=False):
        self.ignore_case = ignore_case
        self.train_df = self._build_train_df()
        self.validation_df = self._build_validation_df()
        self.test_series = self._build_test_series()
        self.test_sentence_series = self._build_test_sentence_series()
        self.uncased_vocab_series = self._build_uncased_vocab_series()

    @cache('train_df')
    def _build_train_df(self):
        res = self.readers.read_train_dataframe()
        res['token'] = res['token'].apply(self.convert_case)
        return res

    @cache('validation_df')
    def _build_validation_df(self):
        res = self.readers.read_validation_dataframe()
        res['token'] = res['token'].apply(self.convert_case)
        return res

    @cache('test_series')
    def _build_test_series(self):
        res = self.readers.read_test_series()
        res = res.apply(self.convert_case)
        return res

    @cache('test_sentence_series')
    def _build_test_sentence_series(self):
        res = self.readers.read_test_sentence_series()
        for i in range(len(res)):
            res[i] = res[i].apply(self.convert_case)
        return res

    @cache('uncased_vocab_series')
    def _build_uncased_vocab_series(self):
        res = self.readers.read_vocab_uncased_series()
        # res = res.apply(self.convert_case)
        return res

    def convert_case(self, x): return x.lower() if self.ignore_case else x
