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

        super().__init__(tile_dir=tile_dir,
                         input_fnames=input_fnames,
                         target_fnames=target_fnames,
                         dataset_config=dataset_config,
                         batch_size=batch_size,
                         image_format=image_format)

        self.mask_fnames = mask_fnames
        # list label_weights: weight for each label
        self.label_weights = None
        if 'label_weights' in dataset_config:
            self.label_weights = dataset_config['label_weights']

    def __getitem__(self, index):
        """
        Get a batch of data. Concatenate mask with target image.
        These will be separated again when computing the loss, it's just
        a backward way of being able to add weights/masks for loss in Keras.

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
        # Whether to normalize outputs
        norm_output = self.model_task != 'segmentation' and self.normalize
        # Loop through batch indices
        input_image = []
        target_image = []
        aug_idx = 0
        for idx in range(start_idx, end_idx, 1):
            cur_input_fnames = self.input_fnames.iloc[self.row_idx[idx]]
            cur_target_fnames = self.target_fnames.iloc[self.row_idx[idx]]
            cur_mask_fnames = self.mask_fnames.iloc[self.row_idx[idx]]

            if self.augmentations:
                aug_idx = np.random.choice([0, 1, 2, 3, 4, 5], 1)
            cur_input = super()._get_volume(
                fname_list=cur_input_fnames.split(','),
                normalize=self.normalize,
                aug_idx=aug_idx,
            )
            cur_target = super()._get_volume(
                fname_list=cur_target_fnames.split(','),
                normalize=norm_output,
                aug_idx=aug_idx,
            )
            cur_mask = super()._get_volume(
                fname_list=cur_mask_fnames.split(','),
                normalize=False,
                aug_idx=aug_idx,
            )
            if self.label_weights is not None:
                wtd_mask = np.zeros(cur_mask.shape)
                for label_idx in range(len(self.label_weights)):
                    wtd_mask += (cur_mask == label_idx) * \
                                self.label_weights[label_idx]
                cur_mask = wtd_mask
            # Concatenate target and mask to one target
            cur_target = np.concatenate((cur_target, cur_mask), axis=0)

            input_image.append(cur_input)
            target_image.append(cur_target)

        input_image = np.stack(input_image)
        target_image = np.stack(target_image)
        return input_image, target_image
