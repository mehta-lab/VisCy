import numpy as np
import os
import pandas as pd

from micro_dl.utils.train_utils import split_train_val_test


class BaseTrainingTable:
    """Generates the training table/info"""

    def __init__(self, df_metadata, input_channels, target_channels,
                 split_by_column, split_ratio, mask_channels=None):
        """Init

        :param pd.DataFrame df_metadata: Dataframe with columns: [channel_num,
         sample_num, timepoint, file_name_0, file_name_1, ......, file_name_n]
        :param list input_channels: list of input channels
        :param list target_channels: list of target channels
        :param str split_by_column: column to be used for train-val-test split
        :param dict split_ratio: dict with keys train, val, test and values are
         the corresponding ratios
        :param list of ints/None mask_channels: Use mask channel if specified
        """

        self.df_metadata = df_metadata
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.split_by_column = split_by_column
        self.split_ratio = split_ratio
        self.mask_channels = mask_channels

    @staticmethod
    def _get_col_name(channel_ids):
        column_names = []
        for c_name in channel_ids:
            cur_fname = 'file_name_{}'.format(c_name)
            column_names.append(cur_fname)
        return column_names

    def _get_df(self, row_idx, retain_columns):
        """
        Get a df from the given row indices and column names/channel_ids
        Merge all input/output file paths into the column fpaths_input/
        fpaths_output.

        :param list row_idx: indices to df_metadata that belong to train, val,
         test splits
        :param list retain_columns: headers of the columns to retain in
         df_metadata
        :return: pd.DataFrame with retain_columns and [fpaths_input,
         fpaths_target]
        """

        input_column_names = self._get_col_name(self.input_channels)
        cur_df = self.df_metadata[row_idx].copy(deep=True)
        cur_df['fpaths_input'] = (
            cur_df[input_column_names].apply(lambda x: ','.join(x), axis=1)
        )
        target_column_names = self._get_col_name(self.target_channels)
        cur_df['fpaths_target'] = (
            cur_df[target_column_names].apply(lambda x: ','.join(x), axis=1)
        )
        if self.mask_channels is not None:
            mask_column_names = self._get_col_name(self.mask_channels)
            cur_df['fpaths_mask'] = (
                cur_df[mask_column_names].apply(lambda x: ','.join(x), axis=1)
            )
        df = cur_df[retain_columns]
        return df

    def train_test_split(self):
        """Split into train-val-test

        :return: pd.DataFrame for train, val and test
        """

        unique_values = self.df_metadata[self.split_by_column].unique()
        # DOES NOT HANDLE NON-INTEGER VALUES. map to int if string
        # the sample_idxs are required for evaluating performance on test set
        assert np.issubdtype(unique_values.dtype, np.integer)
        split_idx = split_train_val_test(
            unique_values, self.split_ratio['train'],
            self.split_ratio['test'], self.split_ratio['val']
        )
        train_set = split_idx['train']
        train_idx = self.df_metadata[self.split_by_column].isin(train_set)
        retain_columns = ['channel_idx', 'pos_idx', 'time_idx', "slice_idx",
                          'fpaths_input', 'fpaths_target']
        if self.mask_channels is not None:
            retain_columns.append('fpaths_mask')

        df_train = self._get_df(train_idx, retain_columns)

        test_set = split_idx['test']
        test_idx = self.df_metadata[self.split_by_column].isin(test_set)
        df_test = self._get_df(test_idx, retain_columns)

        if self.split_ratio['val']:
            val_set = split_idx['val']
            val_idx = self.df_metadata[self.split_by_column].isin(val_set)
            df_val = self._get_df(val_idx, retain_columns)
            return df_train, df_val, df_test, split_idx
        return df_train, df_test, split_idx
