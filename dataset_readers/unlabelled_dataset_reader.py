import pandas as pd


class UnlabelledDatasetReader():
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_to_series(self) -> pd.Series:
        arr = []

        with open(self.file_path, "r") as f:
            for line in f.readlines():
                line = line[:-1]
                if not len(line):
                    continue
                arr.append(line)

        return pd.Series(arr)
