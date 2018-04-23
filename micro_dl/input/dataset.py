"""Dataset classes"""
import numpy as np
import pandas as pd
import keras

from micro_dl.utils.image_utils import resample_image
from micro_dl.utils.train_utils import split_train_val_test


class BaseDataSet(keras.utils.Sequence):
    """Base class for input and target images

    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """

    def __init__(self, input_fnames, target_fnames, batch_size,
                 augmentations=None, shuffle=True, random_seed=42):
        """Init

        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param int batch_size: num of datasets in each batch
        :param dict augmentations: dictionary with allowed augmentations as
         keys and distortion amount as values
        :param bool shuffle: shuffle data for each epoch
        :param int random_seed: initialize the random number generator with
         this seed
        """

        self.input_fnames = input_fnames
        self.target_fnames = target_fnames
        self.batch_size = batch_size

        self.augmentations = augmentations
        self.shuffle = shuffle
        num_samples = len(self.input_fnames)
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.on_epoch_end()

    def __len__(self):
        """Gets the number of batches per epoch"""

        n_batches = int(self.num_samples / self.batch_size)
        return n_batches

    def __getitem__(self, index):
        """Get a batch of data

        https://www.tensorflow.org/performance/performance_guide#use_nchw_imag.
        will use nchw format. shape: [batch_size, num_channels, z, y, x]

        :return: np.ndarrays input_image and target_image of shape
         [batch_size, num_channels, z, y, x]
        """

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx >= self.num_samples:
            end_idx = self.num_samples

        input_image = []
        target_image = []
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            cur_input = self._get_input_volume(cur_input_fnames)
            cur_target = self._get_input_volume(cur_target_fnames)
            input_image.append(cur_input)
            target_image.append(cur_target)
        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        return input_image, target_image

    def on_epoch_end(self):
        """Update indices and shuffle after each epoch"""

        self.row_idx = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.row_idx)

    def __augment_image(self):
        """Augment images based on the augmentations dict"""

        raise NotImplementedError

    def __get_volume(self, fname_list):
        """Read a volume from fname_list

        :param list fname_list: list of file names of input/target images
        :return: np.ndarray of stacked images
        """

        image_volume = []
        for fname in fname_list:
            cur_channel = np.load(fname)
            # add image augmentations here
            if self.augmentations:
                self.__augment_image()
            image_volume.append(cur_channel)

        image_volume = np.stack(image_volume)
        return image_volume

    def _get_input_volume(self, input_fnames):
        """Read one input dataset

        :param str input_fnames: comma separated fnames of the channels used
         as input
        :return: np.ndarray for input dataset
        """

        input_fnames_list = input_fnames.split(',')
        input_volume = self.__get_volume(input_fnames_list)
        return input_volume

    def _get_target_volume(self, target_fnames):
        """Read one target dataset

        :param str input_fnames: comma separated fnames of the channels used
         as target
        :return: np.ndarray for target dataset
        """

        target_fnames_list = target_fnames.split(',')
        target_volume = self.__get_volume(target_fnames_list)
        return target_volume


class BaseTrainingTable:
    """Generates the training table/info"""

    def __init__(self, df_metadata, input_channels, target_channels,
                 split_by_column, split_ratio):
        """Init

        :param pd.DataFrame df_metadata: Dataframe with columns: [channel_num,
         sample_num, timepoint, fname_0, fname_1, ......, fname_n]
        :param list input_channels: list of input channels
        :param list target_channels: list of target channels
        :param str split_by_column: column to be used for train-val-test split
        :param dict split_ratio: dict with keys train, val, test and values are
         the corresponding ratios
        """

        self.df_metadata = df_metadata
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.split_by_column = split_by_column
        self.split_ratio = split_ratio

    def __get_df(self, row_idx, retain_columns):
        """Get a df from the given row indices and column names/channel_ids

        :param list row_idx: indices to df_metadata that belong to train, val,
         test splits
        :param list retain_columns: headers of the columns to retain in
         df_metadata
        :return: pd.DataFrame with retain_columns and [fpaths_input,
         fpaths_target]
        """

        def get_col_name(channel_ids):
            column_names = []
            for c_name in channel_ids:
                cur_fname = 'fname_{}'.format(c_name)
                column_names.append(cur_fname)
            return column_names
        input_column_names = get_col_name(self.input_channels)
        cur_df = self.df_metadata[row_idx].copy(deep=True)
        cur_df['fpaths_input'] = (
            cur_df[input_column_names].apply(lambda x: ','.join(x), axis=1)
        )
        cur_df.drop(columns=input_column_names)

        target_column_names = get_col_name(self.target_channels)
        cur_df['fpaths_target'] = (
            cur_df[target_column_names].apply(lambda x: ','.join(x), axis=1)
        )
        cur_df.drop(columns=target_column_names)
        df = cur_df[retain_columns]
        return df

    def train_test_split(self):
        """Split into train-val-test

        :return: pd.DataFrame for train, val and test
        """

        unique_values = self.df_metadata[self.split_by_column].unique()
        # DOESNOT HANDLE NON-INTEGER VALUES. map to int if string
        assert np.issubdtype(unique_values.dtype, np.integer)
        split_idx = split_train_val_test(
            len(unique_values), self.split_ratio['train'],
            self.split_ratio['test'], self.split_ratio['val']
        )
        train_set = unique_values[split_idx['train']]
        train_idx = self.df_metadata[self.split_by_column].isin(train_set)
        retain_columns = ['channel_num', 'sample_num', 'timepoint',
                          'fpaths_input', 'fpaths_target']
        df_train = self.__get_df(train_idx, retain_columns)

        test_set = unique_values[split_idx['test']]
        test_idx = self.df_metadata[self.split_by_column].isin(test_set)
        df_test = self.__get_df(test_idx, retain_columns)

        if self.split_ratio['val']:
            val_set = unique_values[split_idx['val']]
            val_idx = self.df_metadata[self.split_by_column].isin(val_set)
            df_val = self.__get_df(val_idx, retain_columns)
            return df_train, df_val, df_test
        return df_train, df_test
