"""Croissant 1.1 JSON-LD builder for a packed release tree.

Scans ``<release_root>/data/<dataset_prefix>/{train,test}/*.ozx``, opens
each store via ``iohub`` to read FOV/channel/voxel metadata, and emits a
single Croissant document covering the whole release dataset.
"""

from pathlib import Path
from typing import Any

from dynacell.croissant.static import StaticFields

__all__ = [
    "build_croissant_from_release",
    "merge_croissant_docs",
    "CROISSANT_CONTEXT",
]


CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


# Map from assembly_target → (gene, organelle, channel_label)
_TARGET_META: dict[str, tuple[str, str, str]] = {
    "h2b": ("H2B", "nucleus", "Nuclei"),
    "caax": ("CAAX", "membrane", "Membrane"),
    "sec61b": ("SEC61B", "endoplasmic reticulum", "Structure"),
    "tomm20": ("TOMM20", "mitochondria", "Structure"),
}


def _format_shape(shape_min: list[int], shape_max: list[int]) -> str:
    """Render a store's shape, showing a range on any axis that varies.

    Parameters
    ----------
    shape_min, shape_max : list of int
        Per-axis minimum and maximum over every position in the store.

    Returns
    -------
    str
        e.g. ``"[10, 3, 48, 640, 960]"`` when all positions agree, or
        ``"[7-10, 3, 48, 640, 960]"`` when T varies. Heterogeneous T is by
        design in the A549 pools, so collapsing it to one number would state
        something false about half the FOVs in a published document.
    """
    return "[" + ", ".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in zip(shape_min, shape_max)) + "]"


def _scan_ozx_tree(
    release_root: Path,
    dataset_prefix: str,
) -> list[dict[str, Any]]:
    """Walk ``<release_root>/data/<dataset_prefix>/{train,test}/*.ozx``.

    Opens each OZX via ``iohub.open_ome_zarr`` and extracts root- and
    FOV-level metadata. Returns one dict per store with keys:
    ``file``, ``split``, ``target``, ``condition``, ``n_fov``, ``shape``,
    ``channels``, ``voxel_size``, ``size_bytes``, ``abs_path``.
    """
    import os

    from iohub import open_ome_zarr

    data_dir = release_root / "data" / dataset_prefix
    entries: list[dict[str, Any]] = []
    for split in ("train", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".ozx"):
                continue
            ozx_path = split_dir / fname
            with open_ome_zarr(str(ozx_path), mode="r") as ds:
                root_attrs = dict(ds.zattrs)
                positions = list(ds.positions())
                if not positions:
                    raise ValueError(f"OZX store has no positions: {ozx_path}")
                first_pos_key = positions[0][0]
                # Shape is published as a store-wide claim, so read every
                # position rather than position 0. Heterogeneous T is by design
                # in these pools -- a549-mantis-sec61b-mock train really does
                # hold 14 FOVs at T=7 and 14 at T=10 -- so a single-position
                # probe states one shape for a store where half the FOVs differ.
                shapes = [list(ds[name + "/0"].shape) for name, _ in positions]
                if len({len(sh) for sh in shapes}) != 1:
                    raise ValueError(f"positions disagree on rank in {ozx_path}: {sorted({len(sh) for sh in shapes})}")
                shape_min = [min(sh[a] for sh in shapes) for a in range(len(shapes[0]))]
                shape_max = [max(sh[a] for sh in shapes) for a in range(len(shapes[0]))]
                if shape_min[1] != shape_max[1]:
                    raise ValueError(
                        f"positions disagree on channel count in {ozx_path}: "
                        f"{shape_min[1]}..{shape_max[1]}; channels are a store-level property"
                    )
                omero = ds[first_pos_key].zattrs["ome"]["omero"]
                channels = [ch["label"] for ch in omero["channels"]]
                multiscales = ds[first_pos_key].zattrs["ome"]["multiscales"][0]
                # OZX may carry a {type: scale, scale: [...]} transform,
                # an {type: identity} transform with no spacing, or omit
                # transforms entirely. Take the first scale transform if
                # present; otherwise leave voxel_size None so the doc
                # omits the spacing line rather than emitting wrong values.
                voxel_size: list[float] | None = None
                for tr in multiscales["datasets"][0].get("coordinateTransformations", []):
                    if tr.get("type") == "scale":
                        voxel_size = tr["scale"]  # [T, C, Z, Y, X]
                        break
            stem = fname.removesuffix(".ozx")
            if "_" in stem:
                fallback_target, fallback_condition = stem.split("_", 1)
            else:
                # iPSC convention: <target>.ozx with no perturbation suffix.
                fallback_target, fallback_condition = stem, "none"
            entries.append(
                {
                    "file": fname,
                    "split": split,
                    "target": root_attrs.get("assembly_target", fallback_target.lower()),
                    "condition": root_attrs.get("assembly_condition", fallback_condition),
                    "n_fov": len(positions),
                    "shape_min": shape_min,
                    "shape_max": shape_max,
                    "channels": channels,
                    "voxel_size": voxel_size,
                    "size_bytes": ozx_path.stat().st_size,
                    "abs_path": str(ozx_path),
                }
            )
    return entries


def build_croissant_from_release(
    release_root: Path,
    static: StaticFields,
    dataset_prefix: str = "biohub-a549",
    compute_sha256: bool = False,
) -> dict[str, Any]:
    """Build a Croissant 1.1 JSON-LD by scanning packed OZX stores.

    Derives all metadata directly from the OZX archives via ``iohub`` —
    no dataset manifests or config YAMLs required. Suitable for
    generating Croissant after data has been packed and synced.

    Parameters
    ----------
    release_root
        Root of the release tree (e.g. ``/hpc/projects/.../dynacell_v1``).
        Expects ``data/<dataset_prefix>/{train,test}/*.ozx`` underneath.
    static
        Authoring constants (license, citation, RAI prose).
    dataset_prefix
        Subdirectory under ``data/`` to scan (default ``"biohub-a549"``).
    compute_sha256
        If True, compute SHA-256 of each OZX file. Slow for large
        archives (~400 GB total); default False.

    Returns
    -------
    dict
        JSON-LD-encodable Croissant 1.1 document.
    """
    import hashlib

    entries = _scan_ozx_tree(release_root, dataset_prefix)
    if not entries:
        raise FileNotFoundError(f"No .ozx files found under {release_root / 'data' / dataset_prefix}")

    # Emit s3:// as primary contentUrl (canonical store identifier; what
    # AWS-CLI / boto3 consume directly) and the HTTPS URI in sameAs for
    # browser / curl convenience. Both resolve to identical bytes.
    # Region-explicit hostname (AWS Open Data canonical form, per the datasheet).
    https_base = f"https://{static.aws_bucket}.s3.us-west-2.amazonaws.com/{static.aws_prefix}/data/{dataset_prefix}"
    s3_base = f"s3://{static.aws_bucket}/{static.aws_prefix}/data/{dataset_prefix}"

    file_objects: list[dict[str, Any]] = []
    for e in entries:
        sha256 = ""
        if compute_sha256:
            h = hashlib.sha256()
            with open(e["abs_path"], "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()

        file_objects.append(
            {
                "@type": "cr:FileObject",
                "@id": (f"{dataset_prefix}/{e['split']}/{e['file'].replace('.ozx', '')}"),
                "name": e["file"],
                "description": (
                    f"OZX-packed OME-Zarr for {e['target']} ({e['condition']}, "
                    f"{e['split']} split, {e['n_fov']} FOVs, "
                    f"shape {_format_shape(e['shape_min'], e['shape_max'])})"
                ),
                "contentUrl": f"{s3_base}/{e['split']}/{e['file']}",
                "sameAs": [f"{https_base}/{e['split']}/{e['file']}"],
                "encodingFormat": "application/vnd.ome.zarr+zip",
                "sha256": sha256,
                "contentSize": str(e["size_bytes"]),
            }
        )

    file_objects.append(
        {
            "@type": "cr:FileObject",
            "@id": "dynacell-code-repo",
            "name": "VisCy — DynaCell application code",
            "description": ("Training, prediction, evaluation, and reporting pipelines. BSD-3-Clause license."),
            "contentUrl": "https://github.com/mehta-lab/VisCy/tree/modular-viscy-staging/applications/dynacell",
            "encodingFormat": "application/x-git",
            "license": "https://opensource.org/licenses/BSD-3-Clause",
            "sha256": "",
        }
    )

    demo_key = f"{static.aws_prefix}/demo/dynacell_a549_demo.zip"
    file_objects.append(
        {
            "@type": "cr:FileObject",
            "@id": "dynacell-demo-sample",
            "name": "DynaCell reviewer-accessible sample",
            "description": (
                "Reviewer-accessible sample (<4 GB) with a reduced number of "
                "FOVs and timepoints, intended for quick inspection without "
                "downloading the full release."
            ),
            "contentUrl": f"s3://{static.aws_bucket}/{demo_key}",
            "sameAs": [f"https://{static.aws_bucket}.s3.us-west-2.amazonaws.com/{demo_key}"],
            "encodingFormat": "application/zip",
            "sha256": "",
        }
    )

    targets = sorted({e["target"] for e in entries})
    conditions = sorted({e["condition"] for e in entries})
    total_fov = sum(e["n_fov"] for e in entries)
    total_size_gb = sum(e["size_bytes"] for e in entries) / 1e9
    ref_voxel = entries[0]["voxel_size"]
    voxel_clause = (
        f" Voxel spacing (Z,Y,X) µm/px: {ref_voxel[2]}, {ref_voxel[3]}, {ref_voxel[4]}."
        if ref_voxel is not None
        else ""
    )

    organelles = [_TARGET_META.get(t, (t.upper(), t, t))[1] for t in targets]
    channel_counts = sorted({e["shape_min"][1] for e in entries})
    channel_clause = (
        f" {channel_counts[0]} channels per FOV (label-free Phase3D + brightfield + fluorescence target)."
        if len(channel_counts) == 1
        else f" {channel_counts[0]}–{channel_counts[-1]} channels per FOV "
        "(label-free Phase3D + brightfield + auxiliary / target fluorescence)."
    )

    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": static.name,
        "description": (
            f"{static.name} — paired label-free + fluorescence 3D imaging "
            "for virtual staining and cell profiling. This release "
            f"contains {len(entries)} OZX stores across {len(targets)} "
            f"organelle targets ({', '.join(targets)}) and "
            f"{len(conditions)} condition(s) ({', '.join(conditions)}), "
            f"totalling {total_fov} FOVs ({total_size_gb:.0f} GB)." + channel_clause + voxel_clause
        ),
        "version": "1.1.0",
        "license": static.license_url,
        "citeAs": static.cite_as,
        "url": "https://registry.opendata.aws/dynacell/",
        "isLiveDataset": False,
        "datePublished": "2026",
        "inLanguage": "en",
        # Only modality terms are shared across releases. Cell line and platform
        # come from StaticFields: this builder runs per dataset_prefix, so a
        # hardcoded "A549" would keyword the Allen WTC-11 subset as A549.
        "keywords": [
            "virtual staining",
            "live-cell imaging",
            "label-free microscopy",
            "fluorescence microscopy",
            "3D time-lapse",
            "benchmark",
            "OME-Zarr",
            *static.keywords,
            *sorted(organelles),
            *sorted(conditions),
        ],
        "creator": list(static.creators),
        "publisher": {
            "@type": "sc:Organization",
            "name": static.publisher_name,
            "url": static.publisher_url,
        },
        "funder": {
            "@type": "sc:Organization",
            "name": "Biohub San Francisco",
        },
        "distribution": file_objects,
        "rai:dataCollection": static.rai_data_collection,
        "rai:dataBiases": static.rai_data_biases,
        "rai:annotationsPerItem": static.rai_annotations_per_item,
        "rai:personalSensitiveInformation": static.rai_personal_sensitive_information,
        "rai:dataLimitations": static.rai_data_limitations,
        "rai:dataUseCases": static.rai_data_use_cases,
        "rai:dataSocialImpact": static.rai_data_social_impact,
        "prov:wasDerivedFrom": list(static.prov_was_derived_from),
        "prov:wasGeneratedBy": list(static.prov_was_generated_by),
        "rai:hasSyntheticData": static.rai_has_synthetic_data,
    }


def merge_croissant_docs(
    docs: list[dict[str, Any]],
    *,
    name: str = "DynaCell",
) -> dict[str, Any]:
    """Merge per-dataset Croissant docs into one release-wide document.

    Each input is a complete single-dataset Croissant produced by
    :func:`build_croissant_from_release`. The release ships a single
    ``metadata/croissant.json`` covering subsets with **different licenses**
    (A549 = CC-BY-4.0, iPSC = Allen Institute Terms of Use), so the merge:

    - unions ``distribution``, tagging every per-store ``cr:FileObject``
      (``@id`` contains ``"/"``) with a per-file ``license`` equal to its
      source doc's top-level ``license``;
    - deduplicates shared FileObjects (code repo, demo sample) by ``@id``;
    - sets a release-wide ``name`` and ``license`` (the list of distinct
      per-subset licenses), keeps the superset ``citeAs``, and unions
      ``keywords``;
    - concatenates per-subset RAI prose and ``prov:*`` provenance so no
      subset's statement is dropped.

    Parameters
    ----------
    docs
        Per-dataset Croissant documents, in the order they should appear.
    name
        Release-wide dataset name for the merged document.

    Returns
    -------
    dict
        A single Croissant 1.1 document covering all input datasets.
    """
    import copy

    if not docs:
        raise ValueError("merge_croissant_docs requires at least one document")

    merged = copy.deepcopy(docs[0])

    # distribution: union, per-file license on stores, dedup shared assets.
    seen: set[str] = set()
    distribution: list[dict[str, Any]] = []
    for doc in docs:
        doc_license = doc["license"]
        for file_object in doc["distribution"]:
            file_id = file_object["@id"]
            if file_id in seen:
                continue
            seen.add(file_id)
            file_object = copy.deepcopy(file_object)
            # Per-store OZX FileObjects use "<prefix>/<split>/<name>" ids and
            # inherit their dataset's license; shared assets (code repo, demo)
            # have no "/" and keep any license they already declare.
            if "/" in file_id and "license" not in file_object:
                file_object["license"] = doc_license
            distribution.append(file_object)
    merged["distribution"] = distribution

    merged["name"] = name
    licenses = list(dict.fromkeys(doc["license"] for doc in docs))
    merged["license"] = licenses if len(licenses) > 1 else licenses[0]
    # The most complete citeAs already unions the per-subset references.
    merged["citeAs"] = max((doc["citeAs"] for doc in docs), key=len)
    keywords: list[str] = []
    for doc in docs:
        for keyword in doc.get("keywords", []):
            if keyword not in keywords:
                keywords.append(keyword)
    merged["keywords"] = keywords

    n_stores = sum(1 for fo in distribution if "/" in fo["@id"])
    subset_names = ", ".join(doc["name"] for doc in docs)
    merged["description"] = (
        f"{name} — paired label-free + fluorescence 3D imaging for virtual "
        "staining and cell profiling. Release-wide document covering "
        f"{len(docs)} dataset subsets ({subset_names}) across {n_stores} OZX "
        "stores. Subsets may carry different licenses; see each store's "
        "license field and the per-subset RAI statements below."
    )

    # RAI prose + provenance: concatenate per subset so nothing is dropped.
    for field in (
        "rai:dataCollection",
        "rai:dataBiases",
        "rai:annotationsPerItem",
        "rai:personalSensitiveInformation",
        "rai:dataLimitations",
        "rai:dataUseCases",
        "rai:dataSocialImpact",
    ):
        present = [doc for doc in docs if doc.get(field)]
        distinct = list(dict.fromkeys(doc[field] for doc in present))
        if len(distinct) <= 1:
            merged[field] = distinct[0] if distinct else merged.get(field, "")
        else:
            merged[field] = "\n\n".join(f"{doc['name']}: {doc[field]}" for doc in present)
    merged["rai:hasSyntheticData"] = any(doc.get("rai:hasSyntheticData", False) for doc in docs)
    for field in ("prov:wasDerivedFrom", "prov:wasGeneratedBy"):
        combined: list[Any] = []
        for doc in docs:
            for item in doc.get(field, []):
                if item not in combined:
                    combined.append(item)
        merged[field] = combined

    return merged
