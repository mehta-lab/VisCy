# %%
import collections
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL/")

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import itertools
import unittest


from micro_dl.input.dataset import ToTensor
from micro_dl.training.training import TorchTrainer
from tests.torch_unet.utils.dataset_tests import TestDataset


class TestTraining(unittest.TestCase):
    def SetUp(self):
        # TODO Rewrite
        """
        Initialize different test configurations for TorchTrainer and module
        to run tests on.
        Must build testing environment before running SetUp.
        """
        self.network_config = {
            "model": {
                "architecture": "2.5D",
                "in_channels": 1,
                "out_channels": 1,
                "residual": True,
                "task": "reg",
                "model_dir": None,
            },
            "training": {
                "epochs": 40,
                "learning_rate": 0.00015,
                "optimizer": "adam",
                "loss": "mse",
                "testing_stride": 1,
                "save_model_stride": 1,
                "save_dir": "",
                "mask_type": "unimodal",
                "device": 0,
                "batch_size": 8,
                "data_dir": self.temp,
                "array_name": "arr_0",
                "split_ratio": {
                    "train": 0.66,
                    "test": 0.17,
                    "val": 0.17,
                },
            },
            "dataset": {
                "target_channels": [1],
                "input_channels": [0],
                "window_size": (5, 256, 256),
                "augmentations": None,
            },
        }

    def build_testing_environment(self):
        """
        Create a testing environment:
            create a temp directory
            build zarr data
        """
        dataset_tester = TestDataset()
        dataset_tester.build_zarr_store()

        # sync temp dirs
        self.temp = dataset_tester.temp

    def tearDown(self):
        """
        Clean up temporary file directories created by testing.
        """
        super().tearDown()

    def _random_dataloaders(self, size):
        # TODO Rewrite
        """
        Creates torch dataloaders which load from random normally distributed
        datasets of size 'size'

        :param int size: size of datasets to generate
        :return list dataloaders: list of random pytorch dataloaders for each arch
        """
        assert self.archs, "Must run SetUp first"
        dataloaders = []

        for i in range(len(self.archs)):
            dim = self.data_dims[i]

            tensors = []
            for i in range(size):
                random = ToTensor()(np.random.randn(*dim))
                tensors.append(random.to(torch.device("cuda")))

            dataset = TensorDataset(*tensors)
            dataloaders.append(DataLoader(dataset))
        return dataloaders, dataset

    def _loss_evaluation(self):
        pass

    # -------------- Tests -----------------#

    def test_loss_closeness(self):
        """
        Test functionality of training with gunpowder backend.

        """
        self._all_test_configurations(test="residual")


# %%
