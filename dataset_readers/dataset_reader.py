from typing import List, Tuple
import numpy as np
import pandas as pd


class DatasetReader():
    def __init__(self, file_path: str, sep='\t'):
        self.file_path = file_path
        self.sep = sep

    def read_to_token_dataframe(self) -> pd.DataFrame:
        df = self._init_df()

        for next_df in self.read_to_sentence_dataframe_list():
            df = df.append(next_df)

        return df

    def read_to_sentence_dataframe_list(self) -> List[pd.DataFrame]:
        df_list: List[pd.DataFrame] = []
        rows = []

        with open(self.file_path, "r") as f:
            for line in f.readlines():
                line = line[:-1]
                if not self._is_valid_line(line):
                    df_list.append(
                        pd.DataFrame(rows, columns=['token', 'label'])
                    )
                    rows = []
                    continue
                token, label = self._get_token_label(line)
                rows.append({
                    'token': token,
                    'label': label,
                })

        if len(rows) > 0:
            df_list.append(
                pd.DataFrame(rows, columns=['token', 'label'])
            )

        return df_list

    def _init_df(self):
        return pd.DataFrame(columns=['token', 'label'])

    def _is_valid_line(self, line: str) -> bool:
        return self.sep in line

    def _get_token_label(self, line: str) -> Tuple[str, str]:
        token, label = line.split(self.sep)
        return token, label
