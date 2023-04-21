# %%
import collections
import glob
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools
import os
import unittest
import zarr

import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")

import micro_dl.torch_unet.utils.dataset as dataset_utils
import micro_dl.utils.io_utils as io_utils
import micro_dl.cli.preprocess_script as preprocess_script


class TestDataset(unittest.TestCase):
    def SetUp(self):
        """
        Initialize different test configurations to run tests on
        """
        # generate configuration data
        self.temp = (
            "/hpc/projects/CompMicro/projects/"
            "virtualstaining/torch_microDL/data/testing_temp"
        )
        self.augmentations = {
            "transpose": {
                "transpose_only": [1, 2],
            },
            "mirror": {
                "mirror_only": [1, 2],
            },
            "rotate": {
                "rotation_interval": [0, 3.14],
            },
            "zoom": {
                "scale_interval": [0.7, 1.3],
            },
            "blur": {
                "mode": "gaussian",
                "width_range": [3, 7],
                "sigma": 0.1,
                "prob": 0.2,
                "blur_channels": [0],
            },
            "shear": {
                "angle_range": [-15, 15],
                "prob": 0.2,
                "shear_middle_slice_only": [1, 2],
            },
            "intensity_jitter": {
                "mode": "gaussian",
                "scale_range": [0.7, 1.3],
                "shift_range": [-0.15, 0.15],
                "norm_before_shift": True,
                "jitter_demeaned": True,
                "prob": 0.2,
            },
            "noise": {
                "mode": "gaussian",
                "seed": 14156,
                "clip": True,
                "prob": 0.2,
                "noise_channels": [0],
            },
        }
        self.train_config_augmentations = {
            "augmentations": self.augmentations,
            "split_ratio": {
                "train": 0.66,
                "test": 0.17,
                "val": 0.17,
            },
            "samples_per_epoch": 30,
        }
        self.train_config_raw = {
            "split_ratio": {
                "train": 0.66,
                "test": 0.17,
                "val": 0.17,
            },
            "samples_per_epoch": 30,
        }
        self.all_train_configs = [
            self.train_config_augmentations,
            self.train_config_raw,
        ]

        self.network_config_2d = {"architecture": "2D", "debug_mode": False}
        self.network_config_25d = {"architecture": "2.5D", "debug_mode": False}
        self.all_network_configs = [self.network_config_25d, self.network_config_2d]
        self.dataset_config_2d = {
            "array_name": "arr_0",
            "target_channels": [1],
            "input_channels": [0],
            "window_size": (1, 256, 256),
            "normalization": {
                "scheme": "FOV",
                "type": "median_and_iqr",
            },
            "min_foreground_fraction": 0.2,
            "batch_size": 16,
            "split_ratio": {
                "test": 0.15,
                "train": 0.7,
                "val": 0.15,
            },
        }
        self.dataset_config_25d = {
            "array_name": "arr_0",
            "target_channels": [1],
            "input_channels": [0, 2],
            "window_size": (3, 256, 256),
            "normalization": {
                "scheme": "FOV",
                "type": "median_and_iqr",
            },
            "min_foreground_fraction": 0.2,
            "batch_size": 1,
            "split_ratio": {
                "test": 0.15,
                "train": 0.7,
                "val": 0.15,
            },
        }
        self.all_dataset_configs = [self.dataset_config_25d, self.dataset_config_2d]
        self.preprocessing_config = {
            "zarr_dir": self.temp,
            "preprocessing": {
                "normalize": {
                    "channel_ids": -1,
                    "block_size": 32,
                    "scheme": "FOV",
                    "num_workers": 4,
                },
                "masks": {
                    "channel_ids": -1,
                    "time_ids": -1,
                    "slice_ids": -1,
                    "thresholding_type": "otsu",
                    "output_channel": None,
                    "structuring_element_radius": 5,
                    "num_workers": 4,
                },
            },
        }

    def tearDown(self):
        """
        Cleans up testing environment
        """
        # clean up zarr store
        if os.path.exists(self.zarr_dir):
            shutil.rmtree(self.zarr_dir)

    def build_zarr_store(self, temp, arr_shape, num_positions=5):
        """
        Builds a test zarr store conforming to OME-NGFF HCS Zarr format in the directory
        'temp'.

        Data stored in arrays in this directory (at each position) has the size arr_spatial,
        which denotes the dimensions in:
            (time, channel, z-slice, y, x)

        :param str temp: dir path to build zarr store in
        :param tuple arr_shape: size of dimensions of data
        :param int num_positions: number of positions in array

        """

        writer = io_utils.ZarrWriter(save_dir=temp)
        writer.create_zarr_root(temp)
        self.zarr_dir = writer._ZarrWriter__root_store_path

        for position in range(num_positions):
            writer.init_array(
                position=position,
                data_shape=arr_shape,
                chunk_size=list((1,) * (len(arr_shape) - 2)) + list(arr_shape[-2:]),
                chan_names=[f"Chan_{i}" for i in range(arr_shape[-4])],
                position_name=f"Pos_{position}",
            )
            data = np.random.rand(*arr_shape)
            writer.write(data, p=position)

    def _test_basic_functionality(self):
        """
        Tests functionality with configuration described in self.SetUp().

        Pulls one sample from each dataset created in setup with every key
        corresponding to that dataset (see train_config.array_name)

        :raises AssertionError: Errors if errors found in initiation
        :raises AssertionError: Errors if setting key and accessing dataset produces
                                unexpected behavior
        :raises AssertionError: Errors if samples produced are not of expected size,
                                shape, and type
        """
        try:
            torch_container = dataset_utils.TorchDatasetContainer(
                self.zarr_dir,
                self.train_config,
                self.network_config,
                self.dataset_config,
                device=torch.device("cuda:0"),
            )
        except Exception as e:
            raise AssertionError(f"Failed dataset or container initiation: {e.args}")

        train_dataset = torch_container.train_dataset
        test_dataset = torch_container.test_dataset
        val_dataset = torch_container.val_dataset

        all_datasets = [train_dataset, test_dataset, val_dataset]

        for dataset in all_datasets:
            for key in dataset.data_keys:
                try:
                    dataset.use_key(key)
                except Exception as e:
                    raise AssertionError(f"Error in setting active key: {e.args}")
                try:
                    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                    for sample in dataloader:
                        # ensure sample is a tuple of tensors
                        assert (
                            len(sample) == 2
                        ), f"Target-input tuple shape :{len(sample)} not 2"
                        assert isinstance(
                            sample[0], torch.Tensor
                        ), "Samples produced are not tensors"
                        assert isinstance(
                            sample[1], torch.Tensor
                        ), "Samples produced are not tensors"

                        # ensure sample is correct shape
                        window_size = self.dataset_config["window_size"]
                        batch_size = self.dataset_config["batch_size"]
                        num_target_channels = len(
                            self.dataset_config["target_channels"]
                        )
                        num_input_channels = len(self.dataset_config["input_channels"])

                        # remove extra torch batch  dim
                        input_ = sample[0][0]
                        target_ = sample[1][0]

                        expected_input_size = tuple(
                            [batch_size, num_input_channels] + list(window_size)
                        )
                        assert input_.shape == expected_input_size, (
                            f"Input samples produced of incorrect"
                            f" shape - expected: {expected_input_size} actual: {input_.shape}"
                        )

                        if len(window_size) == 3:
                            expected_target_size = tuple(
                                [batch_size, num_target_channels]
                                + [1]
                                + list(window_size)[1:]
                            )
                        else:
                            expected_target_size = tuple(
                                [batch_size, num_target_channels] + list(window_size)
                            )
                        assert target_.shape == expected_target_size, (
                            f"Target samples produced of incorrect "
                            f" shape - expected: {expected_target_size} actual: {target_.shape}"
                        )
                        break

                except Exception as e:
                    raise AssertionError(
                        f"Error in loading samples from dataloader: {e.args}"
                    )

    def _all_test_configurations(self, test):
        """
        Run specified test on all possible data sampling configurations. Pairs dataset
        and network configurations by index (necessary as there are exclusive parameters).
        With every pair of dataset and network configs, tries every possible training config.

        :param str test: test to run (must be attribute of self)
        """

        for i in range(len(self.all_network_configs)):
            self.dataset_config = self.all_dataset_configs[i]
            self.network_config = self.all_network_configs[i]

            # build basic zarr store
            num_channels = (
                len(self.dataset_config["input_channels"])
                + len(self.dataset_config["target_channels"])
                + 1
            )
            spatial_size = (
                np.array(list(self.dataset_config["window_size"])) * 3
            ).tolist()

            self.build_zarr_store(
                temp=self.temp,
                arr_shape=[1, num_channels] + spatial_size,
            )

            # run preprocessing on that zarr store
            self.preprocessing_config["zarr_dir"] = self.zarr_dir
            preprocess_script.pre_process(self.preprocessing_config)

            # test functionality with each training config
            for train_config in self.all_train_configs:
                self.train_config = train_config

                try:
                    test()
                except Exception as e:
                    # self.tearDown()
                    raise Exception(
                        f"\n\n Exception caught with configuration:"
                        f"\n\n training: {self.train_config}"
                        f"\n\n dataset: {self.dataset_config}"
                        f"\n\n network: {self.network_config}"
                    )

            self.tearDown()

    # ------- tests --------#

    def test_functionality(self):
        """
        Test basic functionality on given configurations:
            - builds zarr store
            - tests container and dataset initation stability
            - tests that sample size, shape, and type matches expected
        """
        self.SetUp()
        self._all_test_configurations(self._test_basic_functionality)
        self.tearDown()


# %%
tester = TestDataset()
#%%
tester.test_functionality()
# %%
