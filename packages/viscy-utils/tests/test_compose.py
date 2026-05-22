import yaml
from pytest import raises

from viscy_utils.compose import deep_merge, load_composed_config


def test_deep_merge_flat():
    """Override replaces base keys, new keys are added."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    assert deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    """Nested dicts are merged recursively, not replaced."""
    base = {"model": {"lr": 0.01, "layers": 3}}
    override = {"model": {"lr": 0.001}}
    result = deep_merge(base, override)
    assert result == {"model": {"lr": 0.001, "layers": 3}}


def test_deep_merge_list_replaces():
    """Lists are replaced entirely, not appended."""
    base = {"channels": ["A", "B"]}
    override = {"channels": ["C"]}
    assert deep_merge(base, override) == {"channels": ["C"]}


def test_deep_merge_does_not_mutate_inputs():
    """Neither base nor override is modified."""
    base = {"model": {"lr": 0.01}}
    override = {"model": {"lr": 0.001}}
    deep_merge(base, override)
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


def test_load_composed_config_resolver_none_identical(tmp_path):
    """resolver=None returns the same dict as omitting the kwarg."""
    (tmp_path / "recipe.yml").write_text(yaml.dump({"model": {"lr": 0.01, "n": 3}}))
    leaf = {"base": ["recipe.yml"], "model": {"lr": 0.001}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    without = load_composed_config(tmp_path / "train.yml")
    with_none = load_composed_config(tmp_path / "train.yml", resolver=None)
    assert without == with_none


def test_load_composed_config_resolver_identity_roundtrip(tmp_path):
    """resolver=lambda d: d leaves the composed dict untouched."""
    (tmp_path / "recipe.yml").write_text(yaml.dump({"model": {"lr": 0.01}}))
    leaf = {"base": ["recipe.yml"], "data": {"batch_size": 4}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    expected = {"model": {"lr": 0.01}, "data": {"batch_size": 4}}
    result = load_composed_config(tmp_path / "train.yml", resolver=lambda d: d)
    assert result == expected


def test_load_composed_config_resolver_sees_post_merge(tmp_path):
    """The resolver receives the fully merged dict with base: already stripped."""
    (tmp_path / "recipe.yml").write_text(yaml.dump({"model": {"lr": 0.01}}))
    leaf = {"base": ["recipe.yml"], "data": {"batch_size": 4}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))

    captured: list[dict] = []

    def capture(d: dict) -> dict:
        captured.append(d)
        return d

    load_composed_config(tmp_path / "train.yml", resolver=capture)
    assert captured == [{"model": {"lr": 0.01}, "data": {"batch_size": 4}}]
    assert "base" not in captured[0]


def test_load_composed_config_resolver_runs_once_nested_base(tmp_path):
    """Nested base: resolution must invoke the resolver exactly once."""
    (tmp_path / "grandparent.yml").write_text(yaml.dump({"a": 1}))
    (tmp_path / "parent.yml").write_text(yaml.dump({"base": ["grandparent.yml"], "b": 2}))
    (tmp_path / "child.yml").write_text(yaml.dump({"base": ["parent.yml"], "c": 3}))

    counter = {"n": 0}

    def count(d: dict) -> dict:
        counter["n"] += 1
        return d

    load_composed_config(tmp_path / "child.yml", resolver=count)
    assert counter["n"] == 1


def test_load_composed_config_strips_underscore_top_level_keys(tmp_path):
    """Top-level keys starting with _ (YAML anchor definitions) are stripped.

    Mirrors the joint-leaf pattern: a YAML merge anchor must be defined at
    top level (the only scope safe_load resolves), but its defining key
    must not survive into the LightningCLI-bound config.
    """
    leaf_yaml = """
_hcs_init_args: &hcs_init_args
  source_channel: [Phase3D]
  batch_size: 4
data:
  init_args:
    children:
      - <<: *hcs_init_args
        data_path: /tmp/a.zarr
      - <<: *hcs_init_args
        data_path: /tmp/b.zarr
"""
    (tmp_path / "train.yml").write_text(leaf_yaml)
    cfg = load_composed_config(tmp_path / "train.yml")
    assert "_hcs_init_args" not in cfg
    assert cfg["data"]["init_args"]["children"][0]["source_channel"] == ["Phase3D"]
    assert cfg["data"]["init_args"]["children"][0]["data_path"] == "/tmp/a.zarr"
    assert cfg["data"]["init_args"]["children"][1]["data_path"] == "/tmp/b.zarr"


def test_load_composed_config_strips_underscore_keys_in_base(tmp_path):
    """Underscore-prefixed top-level keys in base: fragments are also stripped.

    Without per-recursion stripping, a base fragment's anchor definition
    would leak up via deep_merge and pollute the final dict.
    """
    (tmp_path / "recipe.yml").write_text(yaml.dump({"_anchor_def": {"x": 1}, "model": {"y": 2}}))
    leaf = {"base": ["recipe.yml"], "data": {"x": 1}}
    (tmp_path / "train.yml").write_text(yaml.dump(leaf))
    cfg = load_composed_config(tmp_path / "train.yml")
    assert "_anchor_def" not in cfg
    assert cfg == {"model": {"y": 2}, "data": {"x": 1}}


def test_load_composed_config_underscore_strip_runs_after_resolver(tmp_path):
    """Resolver runs before the underscore strip.

    Resolvers see the full dict including any private anchor-def keys.
    The strip is the final step before returning. (Documents the order;
    no current resolver depends on this, but pin the contract.)
    """
    (tmp_path / "train.yml").write_text(yaml.dump({"_x": 1, "data": 2}))
    seen_keys: list[set] = []

    def capture(d: dict) -> dict:
        seen_keys.append(set(d.keys()))
        return d

    cfg = load_composed_config(tmp_path / "train.yml", resolver=capture)
    assert seen_keys == [{"_x", "data"}]  # resolver sees _x
    assert "_x" not in cfg  # but it's stripped from the return value
