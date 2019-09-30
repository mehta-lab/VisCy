import numpy as np


class BaseTrainingTable:
    """Generates the training table/info"""

    def __init__(self,
                 df_metadata,
                 input_channels,
                 target_channels,
                 split_by_column,
                 split_ratio,
                 mask_channels=None,
                 random_seed=None):
        """Init

        :param pd.DataFrame df_metadata: Dataframe with columns: [channel_num,
         sample_num, timepoint, file_name_0, file_name_1, ......, file_name_n]
        :param list input_channels: list of input channels
        :param list target_channels: list of target channels
        :param str split_by_column: column to be used for train-val-test split
         or index of df_metadata
        :param dict split_ratio: dict with keys train, val, test and values are
         the corresponding ratios
        :param list of ints/None mask_channels: Use mask channel if specified
        :param int random_seed: between 0 and uint32, random seed for train-val-test split
        """

        self.df_metadata = df_metadata
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.split_by_column = split_by_column
        self.split_ratio = split_ratio
        self.mask_channels = mask_channels
        self.random_seed = random_seed

    @staticmethod
    def _get_col_name(channel_ids):
        """
        Get file names for given channels

        :param list channel_ids: Channel integer indices
        :return list column_names: Channel file name strings
        """
        column_names = []
        for c_name in channel_ids:
            cur_fname = 'file_name_{}'.format(c_name)
            column_names.append(cur_fname)
        return column_names

    def _get_df(self, cur_df, retain_columns):
        """
        Get a dataframe from the given row indices and column names/channel_ids.
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

    def _get_df_subset(self, split_idx):
        """
        Get a subset of the dataframe given the indices of either
        train, val or test set. Split is done given specified split column.

        :param list split_idx: Integer indices for which to get subset
            of dataframe.
        :return pd.DataFrame df_split: Subset of table for given split
        """
        df_split = self.df_metadata.copy(deep=True)
        if self.split_by_column == 'index':
            df_split = df_split.iloc[split_idx]
        else:
            df_idx = df_split[self.split_by_column].isin(split_idx)
            df_split = df_split[df_idx]
        return df_split

    @staticmethod
    def split_train_val_test(sample_set,
                             train_ratio,
                             test_ratio,
                             val_ratio=None,
                             random_seed=None):
        """
        Generate indices for train, validation and test split

        This can be achieved by using sklearn.model_selection.train_test_split
        twice... :-)

        :param np.array/list sample_set: A set of unique integer indices,
            for split column, not necessarily continuous values
        :param float train_ratio: between 0 and 1, percent of samples to be
            used for training
        :param float test_ratio: between 0 and 1, percent of samples to be
            used for test set
        :param float val_ratio: between 0 and 1, percent of samples to be
            used for the validation set
        :param int random_seed: between 0 and 2**32 - 1, random seed for
            train-val-test split
        :return: dict split_idx with keys [train, val, test] and values as lists
        :raises AssertionError: If ratios don't add up to 1
        """
        assert train_ratio + val_ratio + test_ratio == 1, \
            'train, val and test ratios do not add up to 1'
        num_samples = len(sample_set)
        num_test = int(test_ratio * num_samples)
        num_test = max(num_test, 1)

        np.random.seed(random_seed)
        split_idx = {}
        test_idx = np.random.choice(sample_set, num_test, replace=False)
        split_idx['test'] = test_idx.tolist()
        rem_set = set(sample_set) - set(test_idx)
        rem_set = list(rem_set)

        if val_ratio:
            num_val = int(val_ratio * num_samples)
            num_val = max(num_val, 1)
            val_idx = np.random.choice(rem_set, num_val, replace=False)
            split_idx['val'] = val_idx.tolist()
            rem_set = set(rem_set) - set(val_idx)
            rem_set = list(rem_set)

        train_idx = np.array(rem_set, dtype='int')
        split_idx['train'] = train_idx.tolist()
        return split_idx

    def train_test_split(self):
        """
        Split into train-val-test

        :return: pd.DataFrame for train, val and test
        """

        if self.split_by_column == 'index':
            unique_values = self.df_metadata.index.values.tolist()
        else:
            unique_values = self.df_metadata[
                self.split_by_column].dropna().astype(int).unique().tolist()
        assert len(unique_values) > 1,\
            "{} only contains one or less values, can't be split "\
            "into train/val".format(self.split_by_column)
        # DOES NOT HANDLE NON-INTEGER VALUES. map to int
        # the sample_idxs are required for evaluating performance on test set
        unique_values = np.asarray(unique_values, dtype='uint32')
        split_idx = self.split_train_val_test(
            unique_values,
            self.split_ratio['train'],
            self.split_ratio['test'],
            self.split_ratio['val'],
            self.random_seed,
        )
        # Specify which columns to keep in training tables
        retain_columns = ['channel_idx',
                          'pos_idx',
                          'time_idx',
                          'slice_idx',
                          'fpaths_input',
                          'fpaths_target']
        if self.mask_channels is not None:
            retain_columns.append('fpaths_mask')

        train_set = split_idx['train']
        df_train = self._get_df_subset(train_set)
        df_dict = {'df_train': self._get_df(df_train, retain_columns), }

        test_set = split_idx['test']
        df_test = self._get_df_subset(test_set)
        df_dict['df_test'] = self._get_df(df_test, retain_columns)

        df_val = None
        if self.split_ratio['val']:
            val_set = split_idx['val']
            df_val = self._get_df_subset(val_set)
            df_val = self._get_df(df_val, retain_columns)
        df_dict['df_val'] = df_val
        return df_dict, split_idx
