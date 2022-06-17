import glob
import inspect
import json
import logging
import nose.tools
import os
from testfixtures import TempDirectory

import micro_dl.utils.aux_utils as aux_utils


# Create test metadata table
meta_df = aux_utils.make_dataframe()
channel_idx = 5
time_idx = 6
for s in range(3):
    for p in range(4):
        im_temp = aux_utils.get_im_name(
            channel_idx=channel_idx,
            slice_idx=s,
            time_idx=time_idx,
            pos_idx=p,
        )
        # Now dataframes are assumed to have dir name in them
        meta_row = aux_utils.parse_idx_from_name(im_temp)
        meta_row['dir_name'] = 'temp_dir'
        meta_df = meta_df.append(
            meta_row,
            ignore_index=True,
        )


def test_import_object():
    module_name = 'networks'
    class_name = 'InterpUpSampling2D'
    class_inst = aux_utils.import_object(module_name, class_name)
    nose.tools.assert_true(inspect.isclass(class_inst))
    nose.tools.assert_equal(class_inst.__name__, class_name)


def test_read_config():
    with TempDirectory() as tempdir:
        # The function doesn't care about file format, names just have to start with im_
        test_config = {'param': 10}
        tempdir.write('config.yml', json.dumps(test_config).encode())
        config = aux_utils.read_config(os.path.join(tempdir.path, 'config.yml'))
        nose.tools.assert_dict_equal(config, test_config)


def test_get_row_idx():
    row_idx = aux_utils.get_row_idx(meta_df, time_idx, channel_idx)
    nose.tools.assert_equal(row_idx.all(), True)


def test_get_row_idx_slice():
    aux_utils.parse_idx_from_name(im_temp)
    row_idx = aux_utils.get_row_idx(meta_df, time_idx, channel_idx, slice_idx=1)
    for i, val in row_idx.items():
        if meta_df.iloc[i].slice_idx == 1:
            nose.tools.assert_true(val)
        else:
            nose.tools.assert_false(val)


def test_get_row_idx_slice_pos():
    row_idx = aux_utils.get_row_idx(
        meta_df,
        time_idx,
        channel_idx,
        slice_idx=0,
        pos_idx=3,
    )
    for i, val in row_idx.items():
        if meta_df.iloc[i].slice_idx == 0 and meta_df.iloc[i].pos_idx == 3:
            nose.tools.assert_true(val)
        else:
            nose.tools.assert_false(val)


def test_get_meta_idx():
    pos_idx = aux_utils.get_meta_idx(meta_df, time_idx, channel_idx, 0, 1)
    # This corrresponds to the second row in the dataframe
    nose.tools.assert_equal(pos_idx, 1)


def test_get_im_name():
    im_name = aux_utils.get_im_name(
        time_idx=1,
        channel_idx=2,
        slice_idx=3,
        pos_idx=4,
        extra_field='hej',
        int2str_len=1,
    )
    nose.tools.assert_equal(im_name, 'im_c2_z3_t1_p4_hej.png')


def test_get_sms_im_name():
    im_name = aux_utils.get_sms_im_name(
        time_idx=0,
        channel_name='phase',
        slice_idx=10,
        pos_idx=100,
        extra_field='blub',
        ext='.png',
        int2str_len=3,
    )
    nose.tools.assert_equal(im_name, 'img_phase_t000_p100_z010_blub.png')


def test_get_sms_im_name_nones():
    im_name = aux_utils.get_sms_im_name(
        time_idx=0,
        channel_name=None,
        slice_idx=None,
        pos_idx=10,
        extra_field=None,
        ext='.jpg',
        int2str_len=2,
    )
    nose.tools.assert_equal(im_name, 'img_t00_p10.jpg')


def test_get_im_name_default():
    im_name = aux_utils.get_im_name()
    nose.tools.assert_equal(im_name, 'im.png')


def test_sort_meta_by_channel():
    # Create a df with a second channel idx
    meta_copy = meta_df.copy()
    meta_copy.channel_idx = 4
    meta_copy = meta_copy.append(meta_df, ignore_index=True)
    meta_channel = aux_utils.sort_meta_by_channel(meta_copy)
    col_names = list(meta_channel)
    nose.tools.assert_in('file_name_4', col_names)
    nose.tools.assert_in('file_name_5', col_names)


def validate_metadata_indices():
    metadata_ids, tp_dict = aux_utils.validate_metadata_indices(
        meta_df,
        time_ids=-1,
        channel_ids=-1,
        slice_ids=-1,
        pos_ids=-1,
    )
    nose.tools.assert_list_equal( metadata_ids['channel_ids'].tolist(), [5])
    nose.tools.assert_list_equal(metadata_ids['slice_ids'].tolist(), [0, 1, 2])
    nose.tools.assert_list_equal(metadata_ids['time_ids'].tolist(), [6])
    nose.tools.assert_list_equal(metadata_ids['pos_ids'].tolist(), [0, 1, 2, 3])
    nose.tools.assert_is_none(tp_dict)


def test_init_logger():
    with TempDirectory() as tempdir:
        log_fname = os.path.join(tempdir.path, 'log.txt')
        logger = aux_utils.init_logger('test_log', log_fname, 10)
        nose.tools.assert_equal(logger.__class__, logging.Logger)


def test_make_dataframe():
    test_meta = aux_utils.make_dataframe(3)
    nose.tools.assert_tuple_equal(test_meta.shape, (3, 7))
    nose.tools.assert_list_equal(list(test_meta), aux_utils.DF_NAMES)


def test_read_meta():
    with TempDirectory() as tempdir:
        meta_fname = 'test_meta.csv'
        meta_df.to_csv(os.path.join(tempdir.path, meta_fname))
        test_meta = aux_utils.read_meta(tempdir.path, meta_fname)
        # Only testing file name as writing df changes dtypes
        nose.tools.assert_true(test_meta['file_name'].equals(meta_df['file_name']))


def test_save_tile_meta():
    with TempDirectory() as tempdir:
        meta_copy = meta_df.copy()
        meta_copy['fname_0'] = meta_copy['file_name']
        aux_utils.save_tile_meta(meta_copy, 0, tempdir.path)
        meta_fname = glob.glob(os.path.join(tempdir.path, 'tiles_meta.csv'))
        nose.tools.assert_equal(len(meta_fname), 1)


def test_validate_config():
    config_dict = {
        'a': 5,
        'b': 'roligt',
        'c': None,
    }
    params = ['a', 'b']
    check, msg = aux_utils.validate_config(config_dict, params)
    nose.tools.assert_true(check)
    nose.tools.assert_equal(msg, 'Params absent in network_config: []')


def test_get_channel_axis():
    channel_axis = aux_utils.get_channel_axis('channels_last')
    nose.tools.assert_equal(channel_axis, -1)


def test_get_channel_axis_first():
    channel_axis = aux_utils.get_channel_axis('channels_first')
    nose.tools.assert_equal(channel_axis, 1)


def test_adjust_slice_margins():
    slice_ids = list(range(10))
    adjusted_ids = aux_utils.adjust_slice_margins(slice_ids, 5)
    nose.tools.assert_list_equal(adjusted_ids, list(range(2, 8)))


def test_read_json():
    with TempDirectory() as tempdir:
        valid_json = {
            "a": 5,
            "b": 'test',
        }
        tempdir.write('json_file.json', json.dumps(valid_json).encode())
        json_object = aux_utils.read_json(
            os.path.join(tempdir.path, "json_file.json"),
        )
        nose.tools.assert_equal(json_object, valid_json)


def test_write_json():
    with TempDirectory() as tempdir:
        valid_json = {
            "a": 5,
            "b": 'test',
        }
        tempdir.write('json_file.json', json.dumps(valid_json).encode())
        json_name = glob.glob(os.path.join(tempdir.path, 'json_file.json'))
        nose.tools.assert_equal(len(json_name), 1)


def test_get_sorted_names():
    with TempDirectory() as tempdir:
        # The function doesn't care about file format, names just have to start with im_
        empty_json = {}
        tempdir.write('im_10.json', json.dumps(empty_json).encode())
        tempdir.write('im_5.json', json.dumps(empty_json).encode())
        file_names = aux_utils.get_sorted_names(tempdir.path)
        nose.tools.assert_equal(len(file_names), 2)
        nose.tools.assert_equal(file_names[0], 'im_5.json')
        nose.tools.assert_equal(file_names[1], 'im_10.json')


def test_parse_idx_from_name():
    im_name = 'im_c005_z010_t50000_p020.png'
    meta = aux_utils.parse_idx_from_name(im_name, order="cztp")
    nose.tools.assert_equal(meta["channel_idx"], 5)
    nose.tools.assert_equal(meta["slice_idx"], 10)
    nose.tools.assert_equal(meta["time_idx"], 50000)
    nose.tools.assert_equal(meta["pos_idx"], 20)


def test_parse_idx_different_order():
    im_name = 'im_t50000_z010_p020_c005.png'
    meta = aux_utils.parse_idx_from_name(im_name, order="tzpc")
    nose.tools.assert_equal(meta["channel_idx"], 5)
    nose.tools.assert_equal(meta["slice_idx"], 10)
    nose.tools.assert_equal(meta["time_idx"], 50000)
    nose.tools.assert_equal(meta["pos_idx"], 20)


@nose.tools.raises(AssertionError)
def test_parse_idx_from_name_no_channel():
    file_name = 'img_phase_t500_p400_z300.tif'
    aux_utils.parse_idx_from_name(file_name)


def test_parse_sms_name():
    file_name = 'img_phase_t500_p400_z300.tif'
    channel_names = ['brightfield']
    meta_row = aux_utils.parse_sms_name(file_name, channel_names=channel_names)
    nose.tools.assert_equal(channel_names, ['brightfield', 'phase'])
    nose.tools.assert_equal(meta_row['channel_name'], 'phase')
    nose.tools.assert_equal(meta_row['channel_idx'], 1)
    nose.tools.assert_equal(meta_row['time_idx'], 500)
    nose.tools.assert_equal(meta_row['pos_idx'], 400)
    nose.tools.assert_equal(meta_row['slice_idx'], 300)


def test_parse_sms_name_long_channel():
    file_name = 'img_long_c_name_t001_z002_p003.tif'
    channel_names = []
    meta_row = aux_utils.parse_sms_name(file_name, channel_names=channel_names)
    nose.tools.assert_equal(channel_names, ['long_c_name'])
    nose.tools.assert_equal(meta_row['channel_name'], 'long_c_name')
    nose.tools.assert_equal(meta_row['channel_idx'], 0)
    nose.tools.assert_equal(meta_row['time_idx'], 1)
    nose.tools.assert_equal(meta_row['pos_idx'], 3)
    nose.tools.assert_equal(meta_row['slice_idx'], 2)
