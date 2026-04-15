"""Tests for dynacell.preprocess.config."""

from dynacell.preprocess.config import load_preprocess_config


class TestLoadPreprocessConfig:
    """Tests for load_preprocess_config."""

    def test_loads_existing_yaml(self, tmp_path):
        """Loading an existing YAML returns a dict-like with correct values."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key1: value1\nkey2: 42\n")
        cfg = load_preprocess_config(config_file)
        assert cfg.get("key1") == "value1"
        assert cfg.get("key2") == 42

    def test_nonexistent_path_raises(self, tmp_path):
        """Loading a nonexistent path raises FileNotFoundError."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_preprocess_config(tmp_path / "does_not_exist.yaml")

    def test_get_with_default(self, tmp_path):
        """The .get() interface works with fallback defaults."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("present: hello\n")
        cfg = load_preprocess_config(config_file)
        assert cfg.get("present") == "hello"
        assert cfg.get("missing", "fallback") == "fallback"
