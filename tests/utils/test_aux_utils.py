import json
from pathlib import Path

from viscy.utils.aux_utils import read_config


def test_read_config(tmp_path: Path):
    config = tmp_path / "config.yml"
    # The function doesn't care about file format, names just have to start with im_
    test_config = {"param": 10}
    config.write_text(json.dumps(test_config))
    config = read_config(str(config))
    assert config == test_config
