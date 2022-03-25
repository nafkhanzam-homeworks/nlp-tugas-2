from typing import List

import pandas as pd

from dataset_readers.dataset_reader import DatasetReader


class IgnoreLabelDatasetReader(DatasetReader):
    def read_to_token_series(self):
        df = self.read_to_token_dataframe()
        return df['token']

    def read_to_sentence_series(self) -> List[pd.Series]:
        series_list: List[pd.Series] = []
        series = pd.Series()

        with open(self.file_path, "r") as f:
            for line in f.read().splitlines():
                if not self._is_valid_line(line):
                    series_list.append(series)
                    series = pd.Series()
                    continue
                token, _ = self._get_token_label(line)
                series.append(token)

        if len(series) > 0:
            series_list.append(series)

        return series
