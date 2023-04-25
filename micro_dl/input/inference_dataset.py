import iohub.ngff as ngff
import numpy as np
import os
from torch.utils.data import Dataset
import zarr

import micro_dl.utils.normalize as normalize
import micro_dl.utils.aux_utils as aux_utils


class TorchInferenceDataset(Dataset):
    """
    Based off of torch.utils.data.Dataset:
        - https://pytorch.org/docs/stable/data.html

    Custom dataset class for used for inference. Lightweight, dependent upon IOhub to
    read in data from an NGFF-HCS compatible zarr store and perform inference.

    Predictions are written back to a zarr store inside the model directory, unless
    specified elsewhere.
    """

    def __init__(
        self,
        zarr_dir,
        batch_pred_num,
        sample_depth,
        normalize_inputs,
        norm_type,
        norm_scheme,
        device,
        batched_view=True,
    ):
        """
        Initiate object for selecting and passing data to model for inference.

        :param str zarr_dir: path to zarr store
        :param dict dataset_config: dict object of dataset_config
        :param int batch_pred_num: number of predictions to do simultaneously if doing
                            batch prediction
        :param int sample_depth: depth in z of the stack for a single sample,
                            likely equivalent to z depth of network these samples feed
        :param bool normalize_inputs: whether to normalize samples returned
        :param str norm_type: type of normalization that was used on data in training
        :param str norm_scheme: scheme (breadth) of normalization used in training
        :param torch.device device: device to send samples to before returning
        :param bool strided_view: whether to return a strided view of z-stack as a
                            batch of inputs, defaults to True
        """
        self.zarr_dir = zarr_dir
        self.normalize_inputs = normalize_inputs
        self.norm_type = norm_type
        self.norm_scheme = norm_scheme
        self.device = device

        self.batch_pred_num = batch_pred_num
        self.batch_stack_size = batch_pred_num + sample_depth - 1
        self.sample_depth = sample_depth
        self.channels = None
        self.timesteps = None

        self.batched_view = batched_view
        self.source_position = None
        self.data_plate = ngff.open_ome_zarr(
            store_path=zarr_dir,
            layout="hcs",
            mode="r",
        )

    def __len__(self):
        """Returns the number of valid center slices in position * number of timesteps"""
        output_stack_slices = self.source_position.data.shape[-3] - (
            self.sample_depth - 1
        )
        return output_stack_slices // self.batch_pred_num

    def __getitem__(self, idx):
        """
        Returns the requested channels and slices of the data at the current
        source array.

        Note: idx indexes into a mapping of timestep and center-z-slice. For example
        if timestep 2 and z slice 4 of 10 is requested, idx should be:
            2*10 + 4 = 24


        :param int idx: index in timestep & center-z-slice mapping

        :return torch.Tensor data: requested image stack with strided view
        :return tuple meta: Tuple of metadata about this prediction including
            - int start_z: intended  start of z-range of output for this item
            - list norm_statistics: (optional) list of normalization statistics
                        dicts for each channel in the returned array
        """
        # idx -> time & center idx mapping
        start_z = idx * self.batch_pred_num
        end_z = start_z + self.batch_stack_size
        if isinstance(self.channels, int):
            self.channels = [self.channels]

        # retrieve data from selected channels
        chan_inds = [self.source_position.channel_names.index(c) for c in self.channels]
        data = [
            self.source_position.data[self.timestep, c, start_z:end_z, ...]
            for c in chan_inds
        ]
        data = np.stack(data, 0)

        # normalize
        norm_statistics = [
            self._get_normalization_statistics(c)
            for c in self.channels
            if "mask" not in c
        ]
        if self.normalize_inputs:
            data = self._normalize_multichan(data, norm_statistics)
        # build batched view.
        # NOTE: This can be done with ".as_strided()", but is more
        #       readable this way for minimal cost
        sd, bpn = self.sample_depth, self.batch_pred_num
        data = np.stack([data[:, i : i + sd, ...] for i in range(bpn)], 0)

        # convert and return
        data = aux_utils.ToTensor(self.device)(data)
        return data, start_z, self.batch_pred_num, norm_statistics

    def set_source_array(self, row, col, fov, timestep=None, channels=None):
        """
        Sets the source array in the zarr store at zarr_dir that this
        dataset should pull from when __getitem__ is called.

        :param str/int row_name: row_index of position
        :param str/int col_name: colum index of position
        :param str/int fov_name: field of view index
        :param int timestep: (optional) timestep index to retrieve
        :param tuple(str) channels: (optional) channels to retrieve

        :return tuple shape: shape of expected output from this source
        :return type dtype: dtype of expected output from this source
        """
        row, col, fov = map(str, [row, col, fov])
        self.source_position = self.data_plate[row][col][fov]

        self.timestep = 0
        if timestep:
            self.timesteps = timestep

        channel_ids = tuple(range(self.source_position.data.shape[1]))
        self.channels = [self.data_plate.channel_names[id] for id in channel_ids]
        if channels:
            self.channels = channels

        shape, dtype = self.source_position.data.shape, self.source_position.data.dtype
        return (
            1,
            len(self.channels),
        ) + shape[-3:], dtype

    def _get_normalization_statistics(self, channel_name):
        """
        Gets and returns the normalization statistics stored in the .zattrs of a
        specific position.

        :param str channel_name: name of channel
        """
        if self.norm_scheme == "dataset":
            normalization_metadata = self.data_plate.zattrs["normalization"]
            key = "dataset_statistics"
        else:
            normalization_metadata = self.source_position.zattrs["normalization"]
            key = "fov_statistics"
        return normalization_metadata[channel_name][key]

    def _normalize_multichan(self, data, normalization_meta, denorm=False):
        """
        Given the list normalization meta for a specific multi-channel chunk of
        data whose elements are each dicts of normalization statistics.

        performs normalization on the entire stack as dictated by parameters in
        dataset_config.

        :param np.ndarray data: 4d numpy array (c, z, y, x)
        :param list normalization_meta: list of channel norm statistics for array

        :param np.ndarray normalized_data: normalized 4d numpy array (c,z,y,x)
        """
        all_data = []
        for i, channel_norm_meta in enumerate(normalization_meta):
            channel_data = data[i]
            normed_channel_data = self._normalize(
                channel_data,
                channel_norm_meta,
                denorm=denorm,
            )
            all_data.append(normed_channel_data)

        return np.stack(all_data, axis=0)

    def _normalize(self, data, normalization_meta, denorm=False):
        """
        Given the normalization meta for a specific chunk of data in the format:
        {
            "iqr": some iqr,
            "mean": some mean,
            "median": some median,
            "std": some std
        }

        zscores or un-zscores the data based upon the metadata and 'denorm'

        :param np.ndarray data: 3d un-normalized input data
        :param dict normalization_meta: dictionary of statistics containing precomputed
                                    norm values for dataset and FOV
        :param bool denorm: Whether to apply or revert zscoring on this data

        :return np.ndarray normalized_data: denormed data of input data's shape and type
        """
        norm_type = self.norm_type
        norm_function = normalize.unzscore if denorm else normalize.zscore

        if norm_type == "median_and_iqr":
            normalized_data = norm_function(
                data,
                normalization_meta["median"],
                normalization_meta["iqr"],
            )
        elif norm_type == "mean_and_std":
            normalized_data = norm_function(
                data,
                normalization_meta["mean"],
                normalization_meta["std"],
            )

        return normalized_data
