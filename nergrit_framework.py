from dataset_readers.complete_dataset_reader import CompleteDatasetReader


class NergritFramework():
    def __init__(self):
        self.readers = CompleteDatasetReader(dataset_name="nergrit_ner-grit")
