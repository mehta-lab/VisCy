import json
import os

import nose.tools
from testfixtures import TempDirectory

from viscy.utils.aux_utils import read_config


def test_read_config():
    with TempDirectory() as tempdir:
        # The function doesn't care about file format, names just have to start with im_
        test_config = {"param": 10}
        tempdir.write("config.yml", json.dumps(test_config).encode())
        config = read_config(os.path.join(tempdir.path, "config.yml"))
        nose.tools.assert_dict_equal(config, test_config)
