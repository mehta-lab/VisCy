from dynaclr.evaluation.orchestration.readiness import _storage_location, render_markdown


def test_storage_location_classifies_vast_and_native():
    assert _storage_location("/hpc/projects/organelle_phenotyping/datasets/ds/ds.zarr") == "VAST"
    assert _storage_location("/source/ds/4-assemble/ds_1.zarr") == "native assemble"


def test_render_markdown_shows_readiness_and_paths():
    report = {
        "collection": "test",
        "checked_at": "2026-09-30T12:00:00-07:00",
        "datasets": [
            {
                "dataset": "ds",
                "storage": "native assemble",
                "image_store": "/source/ds.zarr",
                "tracking_store": "/source/tracks.zarr",
                "tracking_exists": True,
                "required_markers": ["PCNA", "Phase3D"],
                "missing_normalization_channels": ["Phase3D"],
                "missing_focus_slice_channels": [],
                "source_stale": False,
                "ai_ready": False,
            }
        ],
    }
    rendered = render_markdown(report)
    assert "| ds | native assemble | PCNA, Phase3D | Phase3D |" in rendered
    assert "AI-ready" in rendered
    assert "`/source/ds.zarr`" in rendered
