import yaml
from pytest import raises

from viscy_utils.compose import _deep_merge, load_composed_config


def test_deep_merge_flat():
    """Override replaces base keys, new keys are added."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    assert _deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    """Nested dicts are merged recursively, not replaced."""
    base = {"model": {"lr": 0.01, "layers": 3}}
    override = {"model": {"lr": 0.001}}
    result = _deep_merge(base, override)
    assert result == {"model": {"lr": 0.001, "layers": 3}}


def test_deep_merge_list_replaces():
    """Lists are replaced entirely, not appended."""
    base = {"channels": ["A", "B"]}
    override = {"channels": ["C"]}
    assert _deep_merge(base, override) == {"channels": ["C"]}


def test_deep_merge_does_not_mutate_inputs():
    """Neither base nor override is modified."""
    base = {"model": {"lr": 0.01}}
    override = {"model": {"lr": 0.001}}
    _deep_merge(base, override)
    assert base == {"model": {"lr": 0.01}}
    assert override == {"model": {"lr": 0.001}}


def test_load_composed_config_no_base(tmp_path):
    """Config without base: key is returned as-is."""
    cfg = {"model": {"lr": 0.01}}
    (tmp_path / "train.yml").write_text(yaml.dump(cfg))
    result = load_composed_config(tmp_path / "train.yml")
    assert result == cfg


def test_load_composed_config_base_null(tmp_path):
    """base: null is treated as no base (empty list)."""
    content = "base: null\nmodel:\n  lr: 0.01\n"
    (tmp_path / "train.yml").write_text(content)
    result = load_composed_config(tmp_path / "train.yml")
    assert result == {"model": {"lr": 0.01}}


def test_load_composed_config_single_base(tmp_path):
    """Single base recipe is merged, then overridden by leaf config."""
    recipe = {"model": {"lr": 0.01, "layers": 3}}
    (tmp_path / "recipe.yml").write_text(yaml.dump(recipe))
    leaf = {"base": ["recipe.yml"], "model": {"lr": 0.001}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    result = load_composed_config(tmp_path / "train.yml")
    assert result == {"model": {"lr": 0.001, "layers": 3}}


def test_load_composed_config_multiple_bases(tmp_path):
    """Multiple bases are merged in order, then overridden by leaf."""
    (tmp_path / "data.yml").write_text(yaml.dump({"data": {"batch_size": 16}}))
    (tmp_path / "model.yml").write_text(yaml.dump({"model": {"lr": 0.01}}))
    leaf = {"base": ["data.yml", "model.yml"], "model": {"lr": 0.001}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    result = load_composed_config(tmp_path / "train.yml")
    assert result == {"data": {"batch_size": 16}, "model": {"lr": 0.001}}


def test_load_composed_config_base_string(tmp_path):
    """base: as a single string (not list) works."""
    (tmp_path / "recipe.yml").write_text(yaml.dump({"model": {"lr": 0.01}}))
    leaf = {"base": "recipe.yml", "model": {"lr": 0.001}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    result = load_composed_config(tmp_path / "train.yml")
    assert result == {"model": {"lr": 0.001}}


def test_load_composed_config_nested_base(tmp_path):
    """Bases can themselves have bases (recursive resolution)."""
    (tmp_path / "grandparent.yml").write_text(yaml.dump({"a": 1, "b": 2}))
    (tmp_path / "parent.yml").write_text(yaml.dump({"base": ["grandparent.yml"], "b": 3}))
    (tmp_path / "child.yml").write_text(yaml.dump({"base": ["parent.yml"], "c": 4}))
    result = load_composed_config(tmp_path / "child.yml")
    assert result == {"a": 1, "b": 3, "c": 4}


def test_load_composed_config_circular_raises(tmp_path):
    """Circular base: references raise ValueError."""
    (tmp_path / "a.yml").write_text(yaml.dump({"base": ["b.yml"]}))
    (tmp_path / "b.yml").write_text(yaml.dump({"base": ["a.yml"]}))
    with raises(ValueError, match="Circular"):
        load_composed_config(tmp_path / "a.yml")


def test_load_composed_config_later_base_wins(tmp_path):
    """When two bases set the same key, the later one wins."""
    (tmp_path / "first.yml").write_text(yaml.dump({"model": {"lr": 0.01}}))
    (tmp_path / "second.yml").write_text(yaml.dump({"model": {"lr": 0.1}}))
    leaf = {"base": ["first.yml", "second.yml"]}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    result = load_composed_config(tmp_path / "train.yml")
    assert result == {"model": {"lr": 0.1}}
