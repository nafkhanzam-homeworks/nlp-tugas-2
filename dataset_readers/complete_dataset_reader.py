from dataset_readers.dataset_reader import DatasetReader
from dataset_readers.ignore_label_dataset_reader import IgnoreLabelDatasetReader
from dataset_readers.unlabelled_dataset_reader import UnlabelledDatasetReader


class CompleteDatasetReader():
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.test_dataset_reader = IgnoreLabelDatasetReader(
            file_path=self._build_path("test_preprocess_masked_label"),
        )
        self.train_dataset_reader = DatasetReader(
            file_path=self._build_path("train_preprocess"),
        )
        self.validation_dataset_reader = DatasetReader(
            file_path=self._build_path("valid_preprocess"),
        )
        self.vocab_uncased_dataset_reader = UnlabelledDatasetReader(
            file_path=self._build_path("vocab_uncased"),
        )
        self.vocab_dataset_reader = UnlabelledDatasetReader(
            file_path=self._build_path("vocab"),
        )

    def read_test_series(self):
        return self.test_dataset_reader.read_to_token_series()

    def read_test_sentence_series(self):
        return self.test_dataset_reader.read_to_sentence_series()

    def read_train_dataframe(self):
        return self.train_dataset_reader.read_to_token_dataframe()

    def read_validation_dataframe(self):
        return self.validation_dataset_reader.read_to_token_dataframe()

    def read_vocab_uncased_series(self):
        return self.vocab_uncased_dataset_reader.read_to_series()

    def read_vocab_series(self):
        return self.vocab_dataset_reader.read_to_series()

    def _build_path(self, filename: str) -> str:
        return f"dataset/{self.dataset_name}/{filename}.txt"
