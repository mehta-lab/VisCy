import numpy as np

from micro_dl.input import BaseDataSet


class DataSetWithMask(BaseDataSet):
    """DataSet class that returns input, target images and sample weights"""

    def __init__(self,
                 tile_dir,
                 input_fnames,
                 target_fnames,
                 mask_fnames,
                 dataset_config,
                 batch_size,
                 image_format='zyx'):
        """Init

        https://stackoverflow.com/questions/44747288/keras-sample-weight-array-error
        https://gist.github.com/andreimouraviev/2642384705034da92d6954dd9993fb4d

        :param str tile_dir: directory containing training image tiles
        :param pd.Series input_fnames: pd.Series with each row containing
         filenames for one input
        :param pd.Series target_fnames: pd.Series with each row containing
         filenames for one target
        :param pd.Series mask_fnames: pd.Series with each row containing
         mask filenames
        :param dict dataset_config: Dataset part of the main config file
        :param int batch_size: num of datasets in each batch
        :param str image_format: Tile shape order: 'xyz' or 'zyx'
        """

        super().__init__(tile_dir,
                         input_fnames,
                         target_fnames,
                         dataset_config,
                         batch_size,
                         image_format)
        self.mask_fnames = mask_fnames
        # list label_weights: weight for each label
        self.label_weights = None
        if 'label_weights' in dataset_config:
            self.label_weights = dataset_config['label_weights']

    def __getitem__(self, index):
        """Get a batch of data

        :param int index: batch index
        :return: np.ndarrays input_image and target_image of shape
         [batch_size, num_channels, z, y, x] and mask_image of shape
         [batch_size, z, x, y] for shape order zyx,
         otherwise [..., x, y, z]
        """

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        if end_idx >= self.num_samples:
            end_idx = self.num_samples

        input_image = []
        target_image = []
        aug_idx = 0
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            cur_mask_fnames = self.mask_fnames.iloc[self.row_idx[idx]]

            if self.augmentations:
                aug_idx = np.random.choice([0, 1, 2, 3, 4, 5], 1)
            cur_input = super()._get_volume(cur_input_fnames.split(','),
                                            aug_idx)
            cur_target = super()._get_volume(cur_target_fnames.split(','),
                                             aug_idx)

            # If target is boolean (segmentation masks), convert to float

            if cur_target.dtype == bool:
                cur_target = cur_target.astype(np.float64)
            if self.normalize:
                cur_input = (cur_input - np.mean(cur_input)) /\
                             np.std(cur_input)
                # Only normalize target if we're dealing with regression
                if self.model_task is not 'segmentation':
                    cur_target = (cur_target - np.mean(cur_target)) /\
                                 np.std(cur_target)

            # the mask is based on sum of channel images
            cur_mask = super()._get_volume(cur_mask_fnames.split(','),
                                           aug_idx)
            if self.label_weights is not None:
                wtd_mask = np.zeros(cur_mask.shape)
                for label_idx in range(len(self.label_weights)):
                    wtd_mask += (cur_mask == label_idx) * \
                                self.label_weights[label_idx]
                cur_mask = wtd_mask
            cur_target = np.concatenate((cur_target, cur_mask), axis=0)

            input_image.append(cur_input)
            target_image.append(cur_target)

        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        return input_image, target_image
