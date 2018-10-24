import json
import nose.tools
from testfixtures import TempDirectory

import micro_dl.utils.aux_utils as aux_utils


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


def test_get_ids_from_imname():
    im_name = 'im_c005_z010_t50000_p020.png'
    df_names = ["channel_idx",
                "slice_idx",
                "time_idx",
                "channel_name",
                "file_name",
                "pos_idx"]

    meta = aux_utils.get_ids_from_imname(im_name, df_names, order="cztp")
    nose.tools.assert_equal(meta["channel_idx"], 5)
    nose.tools.assert_equal(meta["slice_idx"], 10)
    nose.tools.assert_equal(meta["time_idx"], 50000)
    nose.tools.assert_equal(meta["pos_idx"], 20)


def test_get_ids_different_order():
    im_name = 'im_t50000_z010_p020_c005.png'
    df_names = ["channel_idx",
                "slice_idx",
                "time_idx",
                "channel_name",
                "file_name",
                "pos_idx"]

    meta = aux_utils.get_ids_from_imname(im_name, df_names, order="tzpc")
    nose.tools.assert_equal(meta["channel_idx"], 5)
    nose.tools.assert_equal(meta["slice_idx"], 10)
    nose.tools.assert_equal(meta["time_idx"], 50000)
    nose.tools.assert_equal(meta["pos_idx"], 20)
