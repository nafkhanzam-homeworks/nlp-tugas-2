from dataset_readers.complete_dataset_reader import CompleteDatasetReader


class NergritFramework():
    def __init__(self):
        self.readers = CompleteDatasetReader(dataset_name="nergrit_ner-grit")

    def init_data(self):
        self.train_df = self._build_train_df()
        self.validation_df = self._build_validation_df()
        self.test_series = self._build_test_series()
        self.test_sentence_series = self._build_test_sentence_series()
        self.uncased_vocab_series = self._build_uncased_vocab_series()
