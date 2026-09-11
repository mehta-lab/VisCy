"""S3 sync wrapper for AWS Open Data uploads.

Wraps the AWS CLI's ``aws s3 sync`` (out-of-band installation; not a
Python dependency). Default mode is dry-run; explicit ``--no-dry-run``
is required to write — S3 is a shared system and an unintended sync
can move terabytes.
"""

import shutil
import subprocess
from pathlib import Path


def sync_to_s3(
    local_root: Path,
    bucket: str,
    prefix: str,
    *,
    dry_run: bool = True,
    extra_args: list[str] | None = None,
) -> int:
    """Run ``aws s3 sync <local_root> s3://<bucket>/<prefix>``.

    Parameters
    ----------
    local_root
        Local directory to sync.
    bucket
        Target bucket name (without ``s3://``).
    prefix
        Key prefix inside the bucket.
    dry_run
        When True (default), passes ``--dryrun`` to the AWS CLI so the
        command prints what *would* sync without actually transferring
        bytes. Required to be flipped explicitly to ``False`` by the
        caller to avoid accidental writes.
    extra_args
        Additional flags to pass to the AWS CLI (e.g.
        ``["--exclude", "*.tmp"]``).

    Returns
    -------
    int
        AWS CLI exit code. 0 == success.

    Raises
    ------
    RuntimeError
        If the ``aws`` CLI is not on ``$PATH``.
    FileNotFoundError
        If ``local_root`` does not exist or is not a directory.
    """
    if shutil.which("aws") is None:
        raise RuntimeError("aws CLI not on PATH; install via brew/apt (or: pip install awscli) before invoking sync")
    if not local_root.is_dir():
        raise FileNotFoundError(f"local_root must be an existing directory: {local_root}")
    cmd = [
        "aws",
        "s3",
        "sync",
        str(local_root),
        f"s3://{bucket}/{prefix}",
    ]
    if dry_run:
        cmd.append("--dryrun")
    if extra_args:
        cmd.extend(extra_args)
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode
