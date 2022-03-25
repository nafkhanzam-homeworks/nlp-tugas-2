import pandas as pd


class UnlabelledDatasetReader():
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_to_series(self) -> pd.Series:
        series = pd.Series()

        with open(self.file_path, "r") as f:
            for line in f.read().splitlines():
                if not len(line):
                    continue
                series.append(line)

        return series
