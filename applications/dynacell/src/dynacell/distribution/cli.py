"""``distribute`` subcommand entry points.

Wired into the ``dynacell`` top-level dispatcher (``_ARGPARSE_COMMANDS``
in ``dynacell/__main__.py``). The retired ``dynacell-paper`` console
script this originally targeted was removed by the Phase 13 migration.
The dispatcher strips its own subcommand token from ``sys.argv``
before invoking ``main()``, so by the time argparse runs here
``sys.argv[1:]`` holds the distribute-side flags.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Route ``dynacell distribute {pack|sample|checksum}`` to handlers."""
    parser = argparse.ArgumentParser(prog="dynacell distribute")
    sub = parser.add_subparsers(dest="action", required=True)

    pack = sub.add_parser("pack", help="Pack a dataset's assembled zarrs into OZX.")
    pack.add_argument(
        "dataset",
        help="Registered dataset name (e.g. aics-hipsc, a549-mantis-2024_11_07).",
    )
    pack.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Top-level output directory; OZX files land at <root>/<dataset>/<split>/.",
    )
    pack.add_argument(
        "--targets",
        help="Comma-separated subset of targets to pack (default: every target).",
    )
    pack.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to pack (default: train,test).",
    )
    pack.add_argument("--overwrite", action="store_true")

    sample = sub.add_parser(
        "sample",
        help="Pack a small subset (FOV/T-bounded) for reviewer downloads.",
    )
    sample.add_argument("dataset")
    sample.add_argument("--output-root", required=True, type=Path)
    sample.add_argument("--max-fovs", required=True, type=int, dest="fov_limit")
    sample.add_argument("--max-t", required=True, type=int, dest="t_limit")
    sample.add_argument("--targets")
    sample.add_argument("--splits", default="train,test")
    sample.add_argument("--overwrite", action="store_true")

    checksum = sub.add_parser(
        "checksum",
        help=(
            "Print sha256 + path for every .ozx under a directory "
            "(diagnostic; does not emit MANIFEST.json — that's "
            "written by `pack`)."
        ),
    )
    checksum.add_argument("root", type=Path)

    sync = sub.add_parser(
        "sync",
        help="Sync a local directory to an S3 bucket (dry-run by default).",
    )
    sync.add_argument("local_root", type=Path)
    sync.add_argument("--bucket", required=True)
    sync.add_argument("--prefix", required=True)
    sync.add_argument(
        "--no-dry-run",
        action="store_true",
        dest="no_dry_run",
        help="Actually transfer bytes (default: dry-run).",
    )

    verify = sub.add_parser(
        "verify-public",
        help="Head-check every contentUrl in a Croissant doc (anonymous S3 access).",
    )
    verify.add_argument("croissant_path", type=Path)

    args = parser.parse_args(sys.argv[1:])

    if args.action in ("pack", "sample"):
        _do_pack(args)
    elif args.action == "checksum":
        _do_checksum(args)
    elif args.action == "sync":
        _do_sync(args)
    elif args.action == "verify-public":
        _do_verify_public(args)


def _do_pack(args: argparse.Namespace) -> None:
    """Pack (or sample-pack) one dataset; writes per-dataset MANIFEST.json."""
    from dynacell.distribution.manifest import write_pack_manifest
    from dynacell.distribution.ozx import PackMode, pack_dataset

    targets = _split_csv(args.targets)
    splits = _split_csv(args.splits)
    mode: PackMode = "sample" if args.action == "sample" else "all"
    fov_limit = getattr(args, "fov_limit", None)
    t_limit = getattr(args, "t_limit", None)

    results = pack_dataset(
        args.dataset,
        output_root=args.output_root,
        mode=mode,
        fov_limit=fov_limit,
        t_limit=t_limit,
        overwrite=args.overwrite,
        targets=targets,
        splits=splits,
    )

    manifest_path = args.output_root / args.dataset / "MANIFEST.json"
    write_pack_manifest(args.dataset, results, manifest_path)

    for r in results:
        print(f"Packed {r.dst_ozx_path}  sha256={r.sha256[:12]}…  bytes={r.bytes}")
    print(f"Wrote {manifest_path}")


def _do_checksum(args: argparse.Namespace) -> None:
    """Recompute sha256 for every .ozx under a directory; print one line each."""
    import hashlib

    for path in sorted(args.root.rglob("*.ozx")):
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                digest.update(chunk)
        print(f"{digest.hexdigest()}  {path}")


def _do_sync(args: argparse.Namespace) -> None:
    """Run aws s3 sync with explicit dry-run gating."""
    from dynacell.distribution.sync import sync_to_s3

    code = sync_to_s3(
        args.local_root,
        bucket=args.bucket,
        prefix=args.prefix,
        dry_run=not args.no_dry_run,
    )
    if code != 0:
        sys.exit(code)


def _do_verify_public(args: argparse.Namespace) -> None:
    """Head-check every contentUrl; exit non-zero if any URL is not 'ok'."""
    from dynacell.distribution.verify import verify_public

    statuses = verify_public(args.croissant_path)
    bad = []
    for url, status in statuses.items():
        line = f"{status:>20}  {url}"
        print(line)
        if status not in ("ok", "skip (non-s3 URL)"):
            bad.append(url)
    if bad:
        sys.exit(1)


def _split_csv(raw: str | None) -> list[str] | None:
    """Split a comma-separated CLI arg into stripped, non-empty tokens.

    Returns ``None`` when ``raw`` is falsy so downstream call sites can
    distinguish "user asked for everything" from "user passed an empty
    list" — both ``--targets ""`` and an entirely-whitespace value
    collapse to ``None`` (treated as "no filter").
    """
    if not raw:
        return None
    tokens = [tok.strip() for tok in raw.split(",")]
    cleaned = [tok for tok in tokens if tok]
    return cleaned or None
