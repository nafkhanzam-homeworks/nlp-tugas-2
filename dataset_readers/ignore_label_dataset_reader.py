from typing import List
import numpy as np

import pandas as pd

from dataset_readers.dataset_reader import DatasetReader


class IgnoreLabelDatasetReader(DatasetReader):
    def read_to_token_series(self):
        df = self.read_to_token_dataframe()
        return df['token']

    def read_to_sentence_series(self) -> List[pd.Series]:
        series_list: List[pd.Series] = []
        rows = []

        with open(self.file_path, "r") as f:
            for line in f.readlines():
                line = line[:-1]
                if not self._is_valid_line(line):
                    series_list.append(
                        pd.Series(rows)
                    )
                    rows = []
                    continue
                token, _ = self._get_token_label(line)
                rows.append(token)

        if len(rows) > 0:
            series_list.append(
                pd.Series(rows)
            )

        return series_list
