"""``croissant`` subcommand entry points.

Wired into the ``dynacell-paper`` top-level dispatcher.
The dispatcher strips its own subcommand token from ``sys.argv``
before invoking ``main()``, so by the time argparse runs here
``sys.argv[1:]`` holds the croissant-side flags.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    """Route ``dynacell-paper croissant {from-release|validate}`` to handlers."""
    parser = argparse.ArgumentParser(prog="dynacell-paper croissant")
    sub = parser.add_subparsers(dest="action", required=True)

    rel = sub.add_parser(
        "from-release",
        help="Generate Croissant by scanning OZX stores in a release tree.",
    )
    rel.add_argument(
        "--release-root",
        type=Path,
        required=True,
        dest="release_root",
        help="Root of the release tree (expects data/<prefix>/{train,test}/*.ozx).",
    )
    rel.add_argument(
        "--dataset-prefix",
        default="biohub-a549",
        dest="dataset_prefix",
        help="Subdirectory under data/ to scan (default: biohub-a549).",
    )
    rel.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the generated JSON file.",
    )
    rel.add_argument(
        "--compute-sha256",
        action="store_true",
        dest="compute_sha256",
        help="Compute SHA-256 for each OZX file (slow for large archives).",
    )
    rel.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip mlcroissant validation after generation.",
    )

    mrg = sub.add_parser(
        "merge-release",
        help="Generate one release-wide Croissant covering all dataset subsets.",
    )
    mrg.add_argument(
        "--release-root",
        type=Path,
        required=True,
        dest="release_root",
        help="Root of the release tree (expects data/<prefix>/{train,test}/*.ozx).",
    )
    mrg.add_argument(
        "--dataset-prefix",
        action="append",
        dest="dataset_prefixes",
        help="Subdirectory under data/ to scan; repeatable (default: biohub-a549 then aics-hipsc).",
    )
    mrg.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the generated JSON file.",
    )
    mrg.add_argument(
        "--compute-sha256",
        action="store_true",
        dest="compute_sha256",
        help="Compute SHA-256 for each OZX file (slow for large archives).",
    )
    mrg.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip mlcroissant validation after generation.",
    )

    val = sub.add_parser("validate", help="Validate a Croissant JSON-LD doc.")
    val.add_argument("path", type=Path, help="Path to the JSON file.")

    args = parser.parse_args(sys.argv[1:])

    if args.action == "from-release":
        _do_from_release(args)
    elif args.action == "merge-release":
        _do_merge_release(args)
    elif args.action == "validate":
        _do_validate(args)


def _do_from_release(args: argparse.Namespace) -> None:
    """Build Croissant by scanning OZX stores in a release tree."""
    from dynacell.croissant.builder import build_croissant_from_release

    static = _static_fields_for(args.dataset_prefix)
    jsonld = build_croissant_from_release(
        release_root=args.release_root,
        static=static,
        dataset_prefix=args.dataset_prefix,
        compute_sha256=args.compute_sha256,
    )

    if not args.no_validate:
        from dynacell.croissant.validate import validate_croissant

        validate_croissant(jsonld)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonld, sort_keys=True, indent=2) + "\n")
    print(f"Wrote {args.output}")


def _do_merge_release(args: argparse.Namespace) -> None:
    """Merge per-dataset Croissant docs into one release-wide document."""
    from dynacell.croissant.builder import (
        build_croissant_from_release,
        merge_croissant_docs,
    )

    prefixes = args.dataset_prefixes or ["biohub-a549", "aics-hipsc"]
    docs = [
        build_croissant_from_release(
            release_root=args.release_root,
            static=_static_fields_for(prefix),
            dataset_prefix=prefix,
            compute_sha256=args.compute_sha256,
        )
        for prefix in prefixes
    ]
    jsonld = merge_croissant_docs(docs)

    if not args.no_validate:
        from dynacell.croissant.validate import validate_croissant

        validate_croissant(jsonld)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonld, sort_keys=True, indent=2) + "\n")
    print(f"Wrote {args.output} ({len(prefixes)} datasets: {', '.join(prefixes)})")


def _do_validate(args: argparse.Namespace) -> None:
    """Run mlcroissant validation against a path; raises on issues."""
    from dynacell.croissant.validate import validate_croissant

    validate_croissant(args.path)
    print(f"Validated {args.path}")


_CONTACT_EMAIL = "shalin.mehta@czbiohub.org"

# Creator entries shared by every DynaCell release. The per-dataset factories
# add the institution that acquired that subset's data.
_BIOHUB_ORG: dict[str, object] = {
    "@type": "sc:Organization",
    "name": "Biohub San Francisco",
    "url": "https://www.czbiohub.org",
}
_CORRESPONDING_AUTHOR: dict[str, object] = {
    "@type": "sc:Person",
    "name": "Shalin B. Mehta",
    "email": _CONTACT_EMAIL,
    "affiliation": {"@type": "sc:Organization", "name": "Biohub San Francisco"},
}

_DYNACELL_BIBTEX = (
    "@inproceedings{dynacell2026, "
    "title={{DynaCell}: an Evaluation Framework for Dynamic 3D "
    "Virtual Staining of Live Cells}, "
    "author={Kalinin, Alexandr A. and Zheng, Dihan and "
    "Theodoro, Taylla Milena and Ivanov, Ivan and "
    "Hirata-Miyasaki, Eduardo and Lee, See-Chi and Liu, Aofei and "
    "Varra, Sricharan Reddy and Chandler, Talon and Pradeep, Soorya "
    "and Liu, Chad and Leonetti, Manuel D. and Arias, Carolina and "
    "Huang, Bo and Mehta, Shalin B.}, "
    "booktitle={Advances in Neural Information Processing Systems "
    "(NeurIPS) Evaluations and Datasets Track}, year={2026}}"
)

_VIANA_BIBTEX = (
    "@article{viana2023ipsc, "
    "title={Integrated intracellular organization and its variations in "
    "human iPS cells}, "
    "author={Viana, Matheus P. and Chen, Jianxu and Schroeder, "
    "Theo A. and others}, "
    "journal={Nature}, volume={613}, pages={345--354}, year={2023}, "
    "doi={10.1038/s41586-022-05553-z}}"
)

_AICS2018_BIBTEX = (
    "@misc{aics2018hipsc, "
    "title={{hiPSC Single-cell Image Dataset}}, "
    "author={{Allen Institute for Cell Science}}, "
    "year={2018}, "
    "howpublished={\\url{https://www.allencell.org/3d-cell-viewer.html}}, "
    "note={Used under the Allen Institute Terms of Use, "
    "\\url{https://www.allencell.org/terms-of-use.html}}}"
)


def _static_fields_for(dataset_prefix: str):
    """Dispatch to the dataset-specific :class:`StaticFields` factory."""
    if dataset_prefix == "biohub-a549":
        return _a549_static_fields()
    if dataset_prefix == "aics-hipsc":
        return _aics_hipsc_static_fields()
    raise ValueError(
        f"No static fields defined for dataset_prefix={dataset_prefix!r}; add a factory in dynacell/croissant/cli.py"
    )


def _a549_static_fields():
    """``StaticFields`` for the biohub-a549 release dataset.

    Authoring constants (license CC-BY-4.0, citation, RAI prose) reflect
    the in-house Biohub Mantis acquisition. Bucket / prefix and the
    DynaCell BibTeX remain placeholders pending PLAN.md O-1/O-2.
    """
    from dynacell.croissant.static import StaticFields

    return StaticFields(
        name="DynaCell — A549 (Mantis)",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        cite_as=_DYNACELL_BIBTEX,
        keywords=("A549", "Mantis"),
        creators=(
            _BIOHUB_ORG,
            {
                "@type": "sc:Organization",
                "name": "University of California San Francisco — Huang Lab",
                "url": "https://huanglab.ucsf.edu",
            },
            _CORRESPONDING_AUTHOR,
        ),
        publisher_name="AWS Open Data",
        publisher_url="https://registry.opendata.aws",
        contact_email=_CONTACT_EMAIL,
        aws_bucket="dynacell",
        aws_prefix="v1",
        rai_data_collection=(
            "The DynaCell A549 dataset was acquired on the Mantis correlative "
            "light-sheet fluorescence and label-free microscope at Biohub "
            "San Francisco. A549 human lung adenocarcinoma cells (ATCC CCL-185) "
            "were cultured and imaged under three conditions: mock-infected, "
            "Zika virus-infected, and Dengue virus-infected (both BSL-2). "
            "Each field of view is a paired 3D time-lapse acquisition combining "
            "label-free transmitted-light channels (multi-angle phase) and a "
            "single fluorescence target marker per store (H2B / nucleus, CAAX / "
            "membrane, SEC61B / endoplasmic reticulum, or TOMM20 / mitochondria). "
            "Plates were imaged at native cadences of 10, 30, or 120 minutes "
            "within an evaluation window of approximately 5–21 hours post-"
            "infection (hpi) and subsampled to a shared 2-hour grid aligned "
            "to integer hpi (~10 frames per FOV) prior to splitting."
        ),
        rai_data_biases=(
            "The dataset covers a single human cell line (A549) acquired on a "
            "single microscope platform (Mantis correlative light-sheet "
            "fluorescence and label-free). Models trained on this data may not "
            "generalize to other cell types, tissue types, organisms, or "
            "microscope platforms without additional validation. The "
            "perturbation panel covers only Zika and Dengue viruses."
        ),
        rai_annotations_per_item=("Each FOV provides paired label-free and fluorescence volumes as ground truth."),
        rai_personal_sensitive_information=(
            "None. All data derive from an established immortalized cell line "
            "(A549, ATCC CCL-185). No human participants or identifiable "
            "biological material are involved."
        ),
        rai_data_limitations=(
            "A549 is a cancer cell line; its organelle morphology may differ "
            "from non-transformed cells. Each acquired field of view carries a "
            "single fluorescence target marker."
        ),
        rai_data_use_cases=(
            "The dataset is intended for two tasks: (1) virtual staining — "
            "predicting fluorescent organelle channels from label-free "
            "transmitted-light volumes; (2) representation learning of "
            "organelle morphology and dynamics in 3D live cells. Validated use "
            "cases (paired test set, see accompanying paper): zero-shot and "
            "fine-tuned virtual staining of nuclear (H2B), membrane (CAAX), "
            "endoplasmic reticulum (SEC61B), and mitochondrial (TOMM20) "
            "targets in A549 cells under mock, ZIKV, and DENV conditions, "
            "evaluated with PCC, SSIM, and FID. Not validated for: clinical "
            "or diagnostic decision support; automated drug discovery without "
            "independent wet-lab validation; cross-cell-line, cross-organism, "
            "or cross-microscope generalization without retraining or "
            "recalibration; segmentation of structures other than the four "
            "targets above."
        ),
        rai_data_social_impact=(
            "Positive impact: virtual staining reduces phototoxicity and "
            "photobleaching in long-term live-cell imaging by lowering the "
            "required fluorescence dose, enabling experiments that are "
            "currently impossible on light-sensitive samples. Public release "
            "of paired label-free / fluorescence volumes lowers the barrier "
            "to entry for image-translation research in microscopy. Risks: "
            "misuse for clinical diagnosis or pre-clinical decisions without "
            "independent validation on the relevant cell type and imaging "
            "platform; over-interpretation of virtual-stain outputs as "
            "faithful proxies for biological ground truth in quantitative "
            "measurements where small intensity errors matter (e.g., kinetic "
            "flux, single-molecule counting). Mitigations: the dataset is "
            "released as an evaluation benchmark with frozen test splits and "
            "held-out perturbations; full data limitations and bias "
            "statements are published alongside the data; benchmark code, "
            "splits, and metric implementations are open-source so claims "
            "can be reproduced and audited."
        ),
        rai_has_synthetic_data=False,
        prov_was_derived_from=[
            {
                "@id": "https://www.atcc.org/products/ccl-185",
                "prov:label": "A549 cell line (ATCC CCL-185)",
                "description": (
                    "Source cell line for the biohub-a549 release; "
                    "imaging data was acquired in-house at Biohub San "
                    "Francisco on the Mantis correlative light-sheet "
                    "fluorescence and label-free microscope. No external "
                    "image dataset was used as input."
                ),
                "prov:wasAttributedTo": {
                    "@id": "https://www.atcc.org",
                    "prov:label": "American Type Culture Collection",
                },
            },
        ],
        prov_was_generated_by=[
            {
                "@type": "prov:Activity",
                "prov:type": {"@id": "https://www.wikidata.org/wiki/Q4929239"},  # Data Collection
                "prov:label": "Mantis acquisition",
                "description": (
                    "3D time-lapse multi-channel acquisition on the "
                    "Mantis correlative light-sheet fluorescence and "
                    "label-free microscope at Biohub San Francisco. A549 "
                    "human lung adenocarcinoma cells were imaged under "
                    "three conditions (mock, ZIKV, DENV) with a single "
                    "fluorescence target marker per store (H2B / nucleus, "
                    "CAAX / membrane, SEC61B / endoplasmic reticulum, or "
                    "TOMM20 / mitochondria)."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:Agent",
                        "@id": "biohub_research_team",
                        "prov:label": "Biohub San Francisco research team",
                    },
                ],
            },
            {
                "@type": "prov:Activity",
                "prov:label": "DynaCell preprocessing",
                "description": (
                    "Phase reconstruction from multi-angle transmitted-"
                    "light volumes via the open-source waveorder "
                    "pipeline; co-registration of label-free and "
                    "fluorescence channels; per-FOV deskewing and stage-"
                    "position metadata correction; channel-wise "
                    "normalization statistics computed and stored as zarr "
                    "attributes. Train/test partitions are FOV-level and "
                    "frozen at v1.0."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:Agent",
                        "@id": "dynacell_authors",
                        "prov:label": "Kalinin et al., Biohub SF",
                    },
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "https://github.com/mehta-lab/waveorder",
                        "prov:label": "waveorder",
                        "description": ("Open-source phase reconstruction pipeline."),
                    },
                ],
            },
        ],
    )


def _aics_hipsc_static_fields():
    """``StaticFields`` for the aics-hipsc release dataset.

    Reprocessed subset of the Allen Institute hiPSC Single-cell Image
    Dataset (Viana et al., Nature 2023). License is the Allen Institute
    Terms of Use (noncommercial research); citation includes both the
    DynaCell paper and Viana et al. per Allen citation policy.
    """
    from dynacell.croissant.static import StaticFields

    return StaticFields(
        name="DynaCell — iPSC (WTC-11)",
        license_url="https://www.allencell.org/terms-of-use.html",
        cite_as=_DYNACELL_BIBTEX + "\n\n" + _VIANA_BIBTEX + "\n\n" + _AICS2018_BIBTEX,
        keywords=("WTC-11", "iPSC"),
        # The Allen Institute acquired this data; DynaCell reprocessed it. Allen
        # must appear in `creator` -- that is the field aggregators read -- not
        # only in prov:wasDerivedFrom. UCSF Huang Lab is deliberately absent: it
        # contributed to the A549 acquisition, not this subset.
        creators=(
            _BIOHUB_ORG,
            {
                "@type": "sc:Organization",
                "name": "Allen Institute for Cell Science",
                "url": "https://www.allencell.org",
            },
            _CORRESPONDING_AUTHOR,
        ),
        publisher_name="AWS Open Data",
        publisher_url="https://registry.opendata.aws",
        contact_email=_CONTACT_EMAIL,
        aws_bucket="dynacell",
        aws_prefix="v1",
        rai_data_collection=(
            "The DynaCell iPSC subset is reprocessed from the Allen "
            "Institute hiPSC Single-cell Image Dataset (Viana et al., "
            "Nature 2023). The Allen Institute for Cell Science acquired "
            "paired brightfield and 3D confocal fluorescence volumes of "
            "WTC-11 human induced pluripotent stem cells (single donor, "
            "gene-edited line) on a spinning-disk confocal microscope, "
            "with endogenous fluorescent tags marking organelle-specific "
            "proteins. DynaCell selects four targets — H2B / nucleus, "
            "CAAX / cell membrane, SEC61B / endoplasmic reticulum, and "
            "TOMM20 / mitochondria — and reconstructs the corresponding "
            "3D phase channel from the brightfield input via the open-"
            "source waveorder pipeline. 500 FOVs per organelle in the "
            "training split, 100 in the held-out evaluation split."
        ),
        rai_data_biases=(
            "The dataset covers a single human induced pluripotent stem "
            "cell line (WTC-11) acquired on a single platform (Allen "
            "Institute spinning-disk confocal). Models trained on this "
            "data may not generalize to other iPSC lines, differentiated "
            "cell types, primary cells, organisms, or microscope "
            "platforms without additional validation. The endogenous-"
            "tagging strategy targets specific protein loci within each "
            "organelle and may not capture morphological variation "
            "visible with alternative markers."
        ),
        rai_annotations_per_item=(
            "Each FOV provides paired label-free (brightfield + "
            "reconstructed Phase3D) and confocal fluorescence volumes as "
            "ground truth."
        ),
        rai_personal_sensitive_information=(
            "None. All data derive from the WTC-11 hiPSC line, originally "
            "established at the Conklin Laboratory (Gladstone Institutes / "
            "UCSF) and distributed via the Coriell Institute (catalog "
            "GM25256); the Allen Institute used genome editing to insert "
            "endogenous fluorescent tags. The original donor consented to "
            "research use of the derived line; the released image data "
            "contain no genomic sequence or other donor-identifying "
            "information."
        ),
        rai_data_limitations=(
            "Single iPSC line (WTC-11); organelle morphology may differ "
            "from differentiated cell types or other iPSC lines. The "
            "DynaCell subset covers 4 of the broader Allen Institute "
            "organelle target set; users wanting additional markers "
            "should consult the original Allen dataset (linked in "
            "prov:wasDerivedFrom). The Phase3D channel is computationally "
            "reconstructed from brightfield via the open-source waveorder "
            "pipeline; reconstruction quality is bounded by that pipeline."
        ),
        rai_data_use_cases=(
            "The dataset is intended for two tasks: (1) virtual staining "
            "— predicting fluorescent organelle channels (nucleus, "
            "membrane, ER, mitochondria) from label-free transmitted-"
            "light volumes; (2) representation learning of organelle "
            "morphology in 3D iPSCs. Validated use cases (held-out 100-"
            "FOV evaluation split, see accompanying paper): zero-shot "
            "and fine-tuned virtual staining of the four targets above, "
            "evaluated with PCC, SSIM, and FID; representation learning "
            "evaluated against organelle classification benchmarks. Not "
            "validated for: clinical or diagnostic decision support; "
            "cross-cell-line, cross-organism, or cross-microscope "
            "generalization without retraining."
        ),
        rai_data_social_impact=(
            "Positive impact: virtual staining reduces phototoxicity and "
            "photobleaching in long-term live-cell imaging by lowering "
            "the required fluorescence dose. Coupling DynaCell's iPSC "
            "training corpus with the A549 dynamic test set enables "
            "principled evaluation of how well models trained on a "
            "static, paired iPSC dataset generalize to dynamic, "
            "perturbed cancer-cell imaging — a real-world transfer "
            "scenario for stem-cell research. Risks: misuse for clinical "
            "diagnosis or therapeutic-grade decisions about iPSC-derived "
            "products without independent validation; over-interpretation "
            "of virtual-stain outputs as faithful proxies for biological "
            "ground truth in quantitative measurements where small "
            "intensity errors matter. Mitigations: the dataset is "
            "released as an evaluation benchmark with frozen test "
            "splits; full data limitations and bias statements are "
            "published alongside the data; benchmark code, splits, and "
            "metric implementations are open-source so claims can be "
            "reproduced and audited."
        ),
        rai_has_synthetic_data=False,
        prov_was_derived_from=[
            {
                "@id": "https://www.allencell.org/data/hipsc-single-cell.html",
                "prov:label": "Allen Institute hiPSC Single-cell Image Dataset",
                "description": (
                    "Original paired imaging dataset of WTC-11 hiPSC "
                    "lines acquired on the Allen Institute spinning-disk "
                    "confocal microscope; the DynaCell iPSC subset "
                    "reprocesses 4 of the broader Allen target set with "
                    "phase reconstruction added via the open-source "
                    "waveorder pipeline. Citation: Viana, M.P., Chen, "
                    "J., Schroeder, T.A. et al. Integrated intracellular "
                    "organization and its variations in human iPS cells. "
                    "Nature 613, 345-354 (2023). "
                    "https://doi.org/10.1038/s41586-022-05553-z"
                ),
                "sc:license": "https://www.allencell.org/terms-of-use.html",
                "prov:wasAttributedTo": {
                    "@id": "https://www.allencell.org",
                    "prov:label": "Allen Institute for Cell Science",
                },
            },
        ],
        prov_was_generated_by=[
            {
                "@type": "prov:Activity",
                "prov:type": {"@id": "https://www.wikidata.org/wiki/Q4929239"},  # Data Collection
                "prov:label": "Allen Institute hiPSC acquisition",
                "description": (
                    "3D paired brightfield + confocal fluorescence "
                    "acquisition of WTC-11 hiPSC lines on the Allen "
                    "Institute spinning-disk confocal microscope. "
                    "Endogenous fluorescent tags marked organelle-"
                    "specific proteins."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:Agent",
                        "@id": "allen_institute_for_cell_science",
                        "prov:label": "Allen Institute for Cell Science",
                        "description": (
                            "See Viana et al., Nature 2023 "
                            "(https://doi.org/10.1038/s41586-022-05553-z) "
                            "for the canonical reference."
                        ),
                    },
                ],
            },
            {
                "@type": "prov:Activity",
                "prov:label": "DynaCell reprocessing",
                "description": (
                    "Selection of 4 target organelles (nucleus, "
                    "membrane, ER, mitochondria) from the broader Allen "
                    "target set; phase reconstruction from brightfield "
                    "via the open-source waveorder pipeline; per-FOV "
                    "channel-wise normalization statistics computed and "
                    "stored as zarr attributes; train/test split frozen "
                    "at FOV level (500 train, 100 test per organelle)."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:Agent",
                        "@id": "dynacell_authors",
                        "prov:label": "Kalinin et al., Biohub SF",
                    },
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "https://github.com/mehta-lab/waveorder",
                        "prov:label": "waveorder",
                        "description": ("Open-source phase reconstruction pipeline."),
                    },
                ],
            },
        ],
    )
