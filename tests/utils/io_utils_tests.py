import nose
import numpy as np
import os
from testfixtures import TempDirectory
import unittest
import zarr

import micro_dl.utils.io_utils as io_utils


class TestZarrReader(unittest.TestCase):

    def setUp(self):
        """Very basic tests to cover waveorder's zarr writer"""

        self.tempdir = TempDirectory()
        self.input_dir = self.tempdir.path
        self.nbr_pos = 2
        self.nbr_times = 3
        self.nbr_channels = 4
        self.nbr_slices = 10
        self.size_y = 15
        self.size_x = 20
        self.channel_names = ['channel1', 'channel2', 'channel3', 'channel4']
        self.dtype = np.uint16
        self.data_shape = (self.nbr_times,
                           self.nbr_channels,
                           self.nbr_slices,
                           self.size_y,
                           self.size_x)
        self.chunk_size = (1, 1, 1, self.size_y, self.size_x)
        data_array = np.zeros((self.nbr_pos,
                               self.nbr_times,
                               self.nbr_channels,
                               self.nbr_slices,
                               self.size_y,
                               self.size_x),
                              dtype=self.dtype)

        # Write test dataset
        zarr_writer = io_utils.ZarrWriter(
            save_dir=self.input_dir,
        )
        for pos_idx in range(self.nbr_pos):
            data_array[pos_idx, ...] = pos_idx + 50
            zarr_writer.create_zarr_root('test_name_pos{}'.format(pos_idx))
            zarr_writer.init_array(
                position=pos_idx,
                data_shape=self.data_shape,
                chunk_size=self.chunk_size,
                chan_names=self.channel_names,
                dtype=data_array.dtype,
            )
            zarr_writer.write(data_array[pos_idx, ...], p=pos_idx)
        # Instantiate zarr reader
        zarr_path = os.path.join(self.input_dir, 'test_name_pos0.zarr')
        self.zarr_reader = io_utils.ZarrReader(zarr_path)

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertFalse(os.path.isdir(self.input_dir))

    def test_init(self):
        self.assertTrue(self.zarr_reader.width is not 0)
        self.assertTrue(self.zarr_reader.height is not 0)
        self.assertTrue(self.zarr_reader.frames is not 0)
        self.assertTrue(self.zarr_reader.slices is not 0)
        self.assertTrue(self.zarr_reader.channels is not 0)
        self.assertTrue(self.zarr_reader.rows is not None)
        self.assertTrue(self.zarr_reader.columns is not None)
        self.assertTrue(self.zarr_reader.wells is not None)
        self.assertTrue(self.zarr_reader.hcs_meta is not None)

        # Check HCS metadata copy
        meta = self.zarr_reader.hcs_meta
        self.assertTrue('plate' in meta.keys())
        self.assertTrue('well' in meta.keys())
        self.assertEqual(len(meta['well']), self.zarr_reader.get_num_positions())
        self.assertTrue('images' in meta['well'][0])
        self.assertTrue(len(meta['well'][0]['images']) != 0)
        self.assertTrue('path' in meta['well'][0]['images'][0])
        self.assertEqual(meta['well'][0]['images'][0]['path'], 'Pos_000')

    def test_get_image(self):
        im = self.zarr_reader.get_image(
            p=0,
            t=2,
            c=1,
            z=3,
        )
        self.assertEqual(im.mean(), 50.)


class TestZarrWriter(unittest.TestCase):

    def setUp(self):
        """Very basic tests to cover waveorder's zarr writer"""

        self.tempdir = TempDirectory()
        self.write_dir = self.tempdir.path

        self.channel_names = ['test_ch1', 'test_ch2']
        self.zarr_writer = io_utils.ZarrWriter(save_dir=self.write_dir)

        self.data_shape = (3, 2, 5, 10, 15)  # T, C, Z, Y, X
        self.dtype = 'uint16'
        self.data_array = np.random.randint(1, 60000, size=self.data_shape, dtype=self.dtype)
        self.chunk_size = (1, 1, 1, 10, 15)
        self.chan_names = ['State0', 'State1']

    def tearDown(self):
        """
        Tear down temporary folder and file structure
        """
        TempDirectory.cleanup_all()
        self.assertFalse(os.path.isdir(self.write_dir))

    def test_init(self):
        self.assertFalse(self.zarr_writer.verbose)

    @nose.tools.raises(AssertionError)
    def test_open_zarr_root(self):
        self.zarr_writer._open_zarr_root(self.write_dir)

    def test_create_zarr_root(self):
        self.zarr_writer.create_zarr_root('test_name')
        self.assertTrue(isinstance(self.zarr_writer.sub_writer.store['Row_0'], zarr.Group))
        # Check Plate Metadata
        self.assertTrue('plate' in self.zarr_writer.sub_writer.plate_meta)
        self.assertTrue('rows' in self.zarr_writer.sub_writer.plate_meta.get('plate').keys())
        self.assertTrue('columns' in self.zarr_writer.sub_writer.plate_meta.get('plate').keys())
        self.assertTrue('wells' in self.zarr_writer.sub_writer.plate_meta.get('plate').keys())
        self.assertEqual(len(self.zarr_writer.sub_writer.plate_meta.get('plate').get('wells')), 0)
        self.assertEqual(len(self.zarr_writer.sub_writer.plate_meta.get('plate').get('columns')), 0)
        self.assertEqual(len(self.zarr_writer.sub_writer.plate_meta.get('plate').get('rows')), 1)
        # Check Well metadata
        self.assertTrue('well' in self.zarr_writer.sub_writer.well_meta)
        self.assertEqual(len(self.zarr_writer.sub_writer.well_meta.get('well').get('images')), 0)

    def test_init_array(self):
        self.zarr_writer.create_zarr_root('test_name')
        self.zarr_writer.init_array(
            position=3,
            data_shape=self.data_shape,
            chunk_size=self.chunk_size,
            chan_names=self.chan_names,
            dtype=self.dtype,
        )
        self.assertEqual(self.zarr_writer.sub_writer.current_position, 3)
        self.assertEqual(self.zarr_writer.sub_writer.dtype, self.dtype)

    def test_write(self):
        self.zarr_writer.create_zarr_root('test_name')
        self.zarr_writer.init_array(
            position=0,
            data_shape=self.data_shape,
            chunk_size=self.chunk_size,
            chan_names=self.chan_names,
            dtype=self.dtype,
        )
        # Write single index for each channel
        self.zarr_writer.write(self.data_array[0, 0, 0, ...], p=0, t=0, c=0, z=0)
        self.assertTrue(np.array_equal(
            self.zarr_writer.sub_writer.store['Row_0']['Col_0']['Pos_000']['arr_0'][0, 0, 0],
            self.data_array[0, 0, 0, ...]),
        )

    def test_write_data_array(self):
        self.zarr_writer.create_zarr_root('test_name')
        self.zarr_writer.init_array(
            position=0,
            data_shape=self.data_shape,
            chunk_size=self.chunk_size,
            chan_names=self.chan_names,
            dtype=self.dtype,
        )
        self.zarr_writer.write(self.data_array, p=0)
        self.assertTrue(np.array_equal(
            self.zarr_writer.sub_writer.store['Row_0']['Col_0']['Pos_000']['arr_0'],
            self.data_array))
