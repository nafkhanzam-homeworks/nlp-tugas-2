from typing import List
import pandas as pd


class DatasetReader():
    def __init__(self, file_path: str, sep='\t'):
        self.file_path = file_path
        self.sep = sep

    def read_to_token_dataframe(self) -> pd.DataFrame:
        df = self._init_df()

        for next_df in self.read_to_sentence_dataframe_list():
            df.merge(next_df)

        return df

    def read_to_sentence_dataframe_list(self) -> List[pd.DataFrame]:
        df_list: List[pd.DataFrame] = []
        df = self._init_df()

        with open(self.file_path, "r") as f:
            for line in f.read().splitlines():
                if not (self.sep in line):
                    df_list.append(df)
                    df = self._init_df()
                    continue
                token, label = line.split(self.sep)
                df.append({
                    'token': token,
                    'label': label,
                })

        if len(df) > 0:
            df_list.append(df)

        return df_list

    def _init_df():
        return pd.DataFrame(columns=['token', 'label'])
