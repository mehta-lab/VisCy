"""Tests for dynacell.croissant builder + validation.

Exercises the from-release builder against a small synthetic OZX tree
and asserts the generated JSON-LD validates against mlcroissant 1.1.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from dynacell.croissant import (
    StaticFields,
    build_croissant_from_release,
    merge_croissant_docs,
)


def _placeholder_static() -> StaticFields:
    """Sentinel-but-non-empty static fields for tests."""
    return StaticFields(
        name="Test dataset",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        cite_as="@misc{test, title={Test}, year={2026}}",
        keywords=("Test cell line",),
        creators=({"@type": "sc:Organization", "name": "Test creator", "url": "https://example.org"},),
        publisher_name="Test publisher",
        publisher_url="https://example.org",
        contact_email="test@example.org",
        aws_bucket="example-bucket",
        aws_prefix="dynacell/v1",
        rai_data_collection="Test prose for data collection.",
        rai_data_biases="Test prose for biases.",
        rai_annotations_per_item="Test prose for annotations per item.",
        rai_personal_sensitive_information="None.",
        rai_data_limitations="Test prose for limitations.",
        rai_data_use_cases="Test prose for use cases.",
        rai_data_social_impact="Test prose for social impact.",
        rai_has_synthetic_data=False,
        prov_was_derived_from=[
            {
                "@id": "https://example.org/source",
                "prov:label": "Test source",
                "description": "Test source dataset.",
            }
        ],
        prov_was_generated_by=[
            {
                "@type": "prov:Activity",
                "prov:label": "Test activity",
                "description": "Test provenance activity.",
            }
        ],
    )


def _make_synthetic_release(root: Path) -> Path:
    """Build a tiny ``data/biohub-a549/{train,test}/*.ozx`` fixture.

    Writes one OME-Zarr per split + packs each into ``.ozx`` so the
    builder's iohub-driven scan has real archives to read.
    """
    import numpy as np
    from iohub import open_ome_zarr
    from iohub.core.ozx import pack_ozx

    data_dir = root / "data" / "biohub-a549"
    for split in ("train", "test"):
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for marker in ("H2B", "TOMM20"):
            zarr_path = split_dir / f"{marker}_mock.zarr"
            with open_ome_zarr(
                str(zarr_path),
                mode="w-",
                layout="hcs",
                channel_names=["Phase3D", "Brightfield", "Structure"],
            ) as plate:
                pos = plate.create_position("A", "1", "0")
                pos.create_image(
                    "0",
                    np.zeros((2, 3, 4, 8, 8), dtype=np.float32),
                    chunks=(1, 1, 1, 8, 8),
                )
            pack_ozx(zarr_path, split_dir / f"{marker}_mock.ozx")
    return root


def _make_release_tree(root: Path, prefix: str, markers: tuple[str, ...]) -> Path:
    """Build ``data/<prefix>/{train,test}/<marker>.ozx`` fixtures."""
    import numpy as np
    from iohub import open_ome_zarr
    from iohub.core.ozx import pack_ozx

    data_dir = root / "data" / prefix
    for split in ("train", "test"):
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for marker in markers:
            zarr_path = split_dir / f"{marker}.zarr"
            with open_ome_zarr(
                str(zarr_path),
                mode="w-",
                layout="hcs",
                channel_names=["Phase3D", "Brightfield", "Structure"],
            ) as plate:
                pos = plate.create_position("A", "1", "0")
                pos.create_image(
                    "0",
                    np.zeros((2, 3, 4, 8, 8), dtype=np.float32),
                    chunks=(1, 1, 1, 8, 8),
                )
            pack_ozx(zarr_path, split_dir / f"{marker}.ozx")
    return root


_CC_BY = "https://creativecommons.org/licenses/by/4.0/"
_ALLEN = "https://www.allencell.org/terms-of-use.html"


def _allen_static() -> StaticFields:
    """Second-dataset static with a distinct license + divergent RAI prose."""
    kwargs = _placeholder_static().__dict__.copy()
    kwargs["name"] = "iPSC subset"
    kwargs["license_url"] = _ALLEN
    kwargs["keywords"] = ("Other cell line",)
    kwargs["creators"] = ({"@type": "sc:Organization", "name": "Other creator", "url": "https://example.net"},)
    kwargs["rai_data_collection"] = "iPSC-specific collection prose."
    return StaticFields(**kwargs)


class TestStaticFields:
    """StaticFields refuses None / empty-string / empty-list at construction."""

    def test_none_field_raises(self):
        """A None argument raises ValueError naming the offending field."""
        kwargs = _placeholder_static().__dict__.copy()
        kwargs["license_url"] = None
        with pytest.raises(ValueError, match="license_url"):
            StaticFields(**kwargs)

    def test_empty_string_field_raises(self):
        """An empty-string argument raises ValueError naming the field."""
        kwargs = _placeholder_static().__dict__.copy()
        kwargs["aws_bucket"] = ""
        with pytest.raises(ValueError, match="aws_bucket"):
            StaticFields(**kwargs)

    def test_empty_list_field_raises(self):
        """An empty-list argument raises ValueError naming the field."""
        kwargs = _placeholder_static().__dict__.copy()
        kwargs["prov_was_derived_from"] = []
        with pytest.raises(ValueError, match="prov_was_derived_from"):
            StaticFields(**kwargs)


class TestFromRelease:
    """build_croissant_from_release scans an OZX tree and emits valid JSON-LD."""

    def test_missing_release_root_raises(self, tmp_path):
        """Empty release tree → FileNotFoundError, not silent empty doc."""
        with pytest.raises(FileNotFoundError):
            build_croissant_from_release(tmp_path, _placeholder_static(), dataset_prefix="biohub-a549")

    def test_emits_one_file_object_per_ozx_plus_extras(self, tmp_path):
        """4 OZX (2 splits × 2 markers) plus code-repo + demo sample extras."""
        _make_synthetic_release(tmp_path)
        jsonld = build_croissant_from_release(tmp_path, _placeholder_static(), dataset_prefix="biohub-a549")
        ozx_objs = [fo for fo in jsonld["distribution"] if fo.get("@id", "").startswith("biohub-a549/")]
        assert len(ozx_objs) == 4
        ids = {fo["@id"] for fo in ozx_objs}
        assert ids == {
            "biohub-a549/train/H2B_mock",
            "biohub-a549/train/TOMM20_mock",
            "biohub-a549/test/H2B_mock",
            "biohub-a549/test/TOMM20_mock",
        }

    def test_rai_fields_present(self, tmp_path):
        """All 9 RAI / sourcing keys land in the generated doc."""
        _make_synthetic_release(tmp_path)
        jsonld = build_croissant_from_release(tmp_path, _placeholder_static(), dataset_prefix="biohub-a549")
        for key in (
            "rai:dataCollection",
            "rai:dataBiases",
            "rai:annotationsPerItem",
            "rai:personalSensitiveInformation",
            "rai:dataLimitations",
            "rai:dataUseCases",
            "rai:dataSocialImpact",
            "rai:hasSyntheticData",
            "prov:wasGeneratedBy",
            "prov:wasDerivedFrom",
        ):
            assert key in jsonld, f"missing {key}"

    def test_keywords_and_creators_come_from_static_fields(self, tmp_path):
        """Attribution is per-dataset, never hardcoded in the builder.

        The builder runs once per ``dataset_prefix``. A hardcoded cell line or
        creator block would keyword the Allen WTC-11 subset as A549 and credit
        it to the A549 collaborators -- wrong attribution in a license-sensitive
        published artifact.
        """
        _make_release_tree(tmp_path, "biohub-a549", ("H2B_mock", "TOMM20_mock"))
        _make_release_tree(tmp_path, "aics-hipsc", ("cell", "SEC61B"))
        a549 = build_croissant_from_release(tmp_path, _placeholder_static(), dataset_prefix="biohub-a549")
        ipsc = build_croissant_from_release(tmp_path, _allen_static(), dataset_prefix="aics-hipsc")

        assert "Test cell line" in a549["keywords"]
        assert "Test cell line" not in ipsc["keywords"]
        assert "Other cell line" in ipsc["keywords"]
        # Modality terms stay shared.
        assert "virtual staining" in a549["keywords"] and "virtual staining" in ipsc["keywords"]

        assert [c["name"] for c in a549["creator"]] == ["Test creator"]
        assert [c["name"] for c in ipsc["creator"]] == ["Other creator"]

    def test_compute_sha256_populates_field(self, tmp_path):
        """--compute-sha256 fills the sha256 field with a 64-hex digest."""
        _make_synthetic_release(tmp_path)
        jsonld = build_croissant_from_release(
            tmp_path,
            _placeholder_static(),
            dataset_prefix="biohub-a549",
            compute_sha256=True,
        )
        ozx_objs = [fo for fo in jsonld["distribution"] if fo.get("@id", "").startswith("biohub-a549/")]
        for fo in ozx_objs:
            assert len(fo["sha256"]) == 64
            int(fo["sha256"], 16)  # raises if not hex


class TestValidate:
    """Generated JSON-LD passes mlcroissant 1.1 validation."""

    def test_validate_passes_for_from_release(self, tmp_path):
        """Generated from-release Croissant validates."""
        pytest.importorskip("mlcroissant")
        from dynacell.croissant.validate import validate_croissant

        _make_synthetic_release(tmp_path)
        jsonld = build_croissant_from_release(tmp_path, _placeholder_static(), dataset_prefix="biohub-a549")
        validate_croissant(jsonld)


class TestMergeRelease:
    """merge_croissant_docs unions two-license per-dataset docs into one."""

    def _two_docs(self, tmp_path):
        """Two per-dataset docs with distinct licenses over a shared tree."""
        _make_release_tree(tmp_path, "biohub-a549", ("H2B_mock", "TOMM20_mock"))
        _make_release_tree(tmp_path, "aics-hipsc", ("cell", "SEC61B"))
        doc_a = build_croissant_from_release(tmp_path, _placeholder_static(), dataset_prefix="biohub-a549")
        doc_b = build_croissant_from_release(tmp_path, _allen_static(), dataset_prefix="aics-hipsc")
        return doc_a, doc_b

    def test_unions_stores_with_per_file_license(self, tmp_path):
        """Every store keeps its own dataset's license after the union."""
        doc_a, doc_b = self._two_docs(tmp_path)
        merged = merge_croissant_docs([doc_a, doc_b])
        assert merged["name"] == "DynaCell"
        stores = [fo for fo in merged["distribution"] if "/" in fo["@id"]]
        assert len(stores) == 8  # 4 per dataset (2 splits × 2 markers)
        for fo in stores:
            expected = _CC_BY if fo["@id"].startswith("biohub-a549/") else _ALLEN
            assert fo["license"] == expected

    def test_dedups_shared_assets(self, tmp_path):
        """Code-repo and demo FileObjects appear once, not once per dataset."""
        doc_a, doc_b = self._two_docs(tmp_path)
        ids = [fo["@id"] for fo in merge_croissant_docs([doc_a, doc_b])["distribution"]]
        assert ids.count("dynacell-code-repo") == 1
        assert ids.count("dynacell-demo-sample") == 1

    def test_top_level_license_lists_both(self, tmp_path):
        """The dataset-level license is the list of distinct subset licenses."""
        doc_a, doc_b = self._two_docs(tmp_path)
        assert merge_croissant_docs([doc_a, doc_b])["license"] == [_CC_BY, _ALLEN]

    def test_rai_prose_concatenated_only_when_divergent(self, tmp_path):
        """Divergent RAI prose is labeled per subset; identical prose stays single."""
        doc_a, doc_b = self._two_docs(tmp_path)
        merged = merge_croissant_docs([doc_a, doc_b])
        assert "Test dataset:" in merged["rai:dataCollection"]
        assert "iPSC subset:" in merged["rai:dataCollection"]
        assert merged["rai:dataBiases"] == "Test prose for biases."

    def test_merged_validates(self, tmp_path):
        """The merged two-license document passes mlcroissant validation."""
        pytest.importorskip("mlcroissant")
        from dynacell.croissant.validate import validate_croissant

        doc_a, doc_b = self._two_docs(tmp_path)
        validate_croissant(merge_croissant_docs([doc_a, doc_b]))


def test_builder_imports_without_the_croissant_extra():
    """Importing the builder must not drag in the optional ``croissant`` extra.

    ``validate.py`` imports mlcroissant at module top, so re-exporting
    ``validate_croissant`` from the package ``__init__`` made an optional extra a
    hard requirement for ``builder`` -- defeating cli.py's lazy imports and the
    --no-validate flag, and erroring this whole file at collection wherever the
    extra is absent (which is every interpreter on this machine).

    Run in a subprocess so a module already imported by the test session cannot
    mask the regression.
    """
    subprocess.run(
        [sys.executable, "-c", "import dynacell.croissant.builder"],
        check=True,
        capture_output=True,
    )
