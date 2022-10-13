import pandas as pd
from os.path import join
from torch.utils.data import Dataset


class csvDataset(Dataset):

    def __init__(self, split: str, path: str):
        self._dset = split
        self._data_path = join(path, '{}.csv'.format(split))
        self._data_df = pd.read_csv(self._data_path)

    def __len__(self):
        return len(self._data_df)

    def __getitem__(self, i):
        target = self._data_df.loc[i, 'target']
        source = self._data_df.loc[i, 'source']
        article_id = self._data_df.loc[i, 'article_id']

        return source, target, article_id