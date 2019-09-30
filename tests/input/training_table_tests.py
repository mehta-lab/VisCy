import nose.tools
import numpy as np
import os
import pandas as pd
import unittest

import micro_dl.input.training_table as training_table
import micro_dl.utils.aux_utils as aux_utils


class TestTrainingTable(unittest.TestCase):

    def setUp(self):
        """
        Set up a dataframe for training table
        """
        # Start frames meta file
        self.meta_name = 'frames_meta.csv'
        self.frames_meta = aux_utils.make_dataframe()
        self.time_ids = [3, 4, 5]
        self.pos_ids = [7, 8, 10, 12, 15]
        self.channel_ids = [0, 1, 2, 3]
        self.slice_ids = [0, 1, 2, 3, 4, 5]
        # Tiles will typically be split into image subsections
        # but it doesn't matter for testing
        for c in self.channel_ids:
            for p in self.pos_ids:
                for z in self.slice_ids:
                    for t in self.time_ids:
                        im_name = aux_utils.get_im_name(
                            channel_idx=c,
                            slice_idx=z,
                            time_idx=t,
                            pos_idx=p,
                        )
                        self.frames_meta = self.frames_meta.append(
                            aux_utils.parse_idx_from_name(im_name),
                            ignore_index=True,
                        )
        self.tiles_meta = aux_utils.sort_meta_by_channel(self.frames_meta)
        self.input_channels = [0, 2]
        self.target_channels = [3]
        self.mask_channels = [1]
        self.split_ratio = {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2,
        }
        # Instantiate class
        self.table_inst = training_table.BaseTrainingTable(
            df_metadata=self.tiles_meta,
            input_channels=self.input_channels,
            target_channels=self.target_channels,
            split_by_column='pos_idx',
            split_ratio=self.split_ratio,
            mask_channels=[1],
            random_seed=42,
        )

    def test__init__(self):
        col_names = ['index', 'channel_idx', 'slice_idx', 'time_idx',
                      'channel_name', 'file_name_0', 'pos_idx',
                      'file_name_1', 'file_name_2', 'file_name_3']

        self.assertListEqual(list(self.table_inst.df_metadata), col_names)
        self.assertListEqual(
            self.table_inst.input_channels,
            self.input_channels,
        )
        self.assertListEqual(
            self.table_inst.target_channels,
            self.target_channels,
        )
        self.assertListEqual(
            self.table_inst.mask_channels,
            self.mask_channels,
        )
        self.assertEqual(self.table_inst.split_by_column, 'pos_idx')
        self.assertDictEqual(self.table_inst.split_ratio, self.split_ratio)
        self.assertEqual(self.table_inst.random_seed, 42)

    def test_get_col_name(self):
        col_names = self.table_inst._get_col_name([1, 3])
        self.assertListEqual(col_names, ['file_name_1', 'file_name_3'])

    def test_get_df(self):
        retain_columns = ['channel_idx',
                          'pos_idx',
                          'time_idx',
                          'slice_idx',
                          'fpaths_input',
                          'fpaths_target',
                          'fpaths_mask']
        concat_df = self.table_inst._get_df(
            cur_df=self.tiles_meta,
            retain_columns=retain_columns,
        )
        self.assertListEqual(list(concat_df), retain_columns)
        # channel_idx is useless once the df is concatenated by channels
        self.assertListEqual(
            concat_df['channel_idx'].unique().tolist(),
            [0],
        )
        self.assertListEqual(
            concat_df['time_idx'].unique().tolist(),
            self.time_ids,
        )
        self.assertListEqual(
            concat_df['slice_idx'].unique().tolist(),
            self.slice_ids,
        )
        self.assertListEqual(
            concat_df['time_idx'].unique().tolist(),
            self.time_ids,
        )
        self.assertListEqual(
            concat_df['pos_idx'].unique().tolist(),
            self.pos_ids,
        )
        # Validate first row of file paths
        row0 = concat_df.iloc[0]
        self.assertEqual(
            row0['fpaths_input'],
            'im_c000_z000_t003_p007.png,im_c002_z000_t003_p007.png',
        )
        self.assertEqual(
            row0['fpaths_target'],
            'im_c003_z000_t003_p007.png',
        )
        self.assertEqual(
            row0['fpaths_mask'],
            'im_c001_z000_t003_p007.png',
        )

    def test_get_df_subset(self):
        subset_df = self.table_inst._get_df_subset([10, 15])
        self.assertListEqual(
            subset_df['pos_idx'].unique().tolist(),
            [10, 15],
        )
        # Should be 2/5 of dataset
        self.assertEqual(subset_df.shape[0] / self.tiles_meta.shape[0], 0.4)

    def test_get_df_subset_index(self):
        self.table_inst.split_by_column = 'index'
        subset_df = self.table_inst._get_df_subset([10, 15, 20])
        self.assertListEqual(
            subset_df.index.values.tolist(),
            ['10', '15', '20'],
        )
        # Should be 2/5 of dataset
        self.assertEqual(subset_df.shape[0], 3)

    def test_split_train_val_test(self):
        split_idx = self.table_inst.split_train_val_test(
            sample_set=self.pos_ids,
            train_ratio=0.4,
            test_ratio=0.4,
            val_ratio=0.2,
        )
        self.assertEqual(len(split_idx['train']), 2)
        self.assertEqual(len(split_idx['val']), 1)
        self.assertEqual(len(split_idx['test']), 2)

    def test_train_test_split(self):
        df_dict, split_idx = self.table_inst.train_test_split()
        # Split train/val/test .6/.2/.2 -> 3/1/1
        self.assertEqual(len(split_idx['train']), 3)
        self.assertEqual(len(split_idx['val']), 1)
        self.assertEqual(len(split_idx['test']), 1)
        # Assert tables are the right size
        total_samples = self.tiles_meta.shape[0]
        self.assertEqual(df_dict['df_train'].shape[0] / total_samples, 0.6)
        self.assertEqual(df_dict['df_val'].shape[0] / total_samples, 0.2)
        self.assertEqual(df_dict['df_test'].shape[0] / total_samples, 0.2)
