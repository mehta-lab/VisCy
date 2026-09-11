"""Public-URL smoke for committed Croissant ``contentUrl`` fields.

NeurIPS 2026 E&D submissions risk desk-rejection when Croissant URLs
do not resolve at submission time. ``verify_public`` issues an
unauthenticated ``head_object`` against every URL in a Croissant doc
to catch 403/404 before reviewers do.
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _iter_content_urls(jsonld: dict[str, Any]) -> list[str]:
    """Walk a Croissant JSON-LD doc and yield every ``contentUrl`` string.

    Parameters
    ----------
    jsonld : dict
        Parsed Croissant document.

    Returns
    -------
    list of str
        Every ``contentUrl`` found under ``distribution``.

    Raises
    ------
    KeyError
        If the doc has no ``distribution`` key. Defaulting to ``[]`` made a
        missing, renamed or ``@graph``-wrapped key indistinguishable from a
        doc with nothing to check: ``verify-public`` then printed nothing and
        exited 0, reporting success having verified no URL at all. Every
        producer in this codebase sets ``distribution`` unconditionally
        (``builder.py`` and ``merge_croissant_docs``), so its absence means
        the doc is not what this tool expects.
    """
    urls: list[str] = []
    distribution = jsonld["distribution"]
    if isinstance(distribution, list):
        for item in distribution:
            if isinstance(item, dict):
                url = item.get("contentUrl")
                if isinstance(url, str):
                    urls.append(url)
    return urls


def verify_public(croissant_path: Path) -> dict[str, str]:
    """Head-check every ``s3://`` URL in a Croissant doc; return a status map.

    Parameters
    ----------
    croissant_path
        Path to a Croissant JSON-LD file
        (e.g. ``preprocessing/dynacell-a549/4-pack-ozx/croissant_a549.json``).

    Returns
    -------
    dict[str, str]
        ``{url: status}``. Documented status values:

        - ``"ok"`` — head_object returned 200.
        - ``"skip (non-s3 URL)"`` — non-``s3://`` URL (https links to
          landing pages, etc.); not exercised here.
        - ``"missing-bucket"`` — malformed ``s3://`` URL (no bucket
          host or empty key).
        - Any AWS error code from
          ``ClientError.response["Error"]["Code"]`` (commonly
          ``"404"``, ``"403"``, ``"NoSuchKey"``, ``"AccessDenied"``).
        - Any unexpected exception's class name as a fallback.

        Caller checks for non-``"ok"``/non-``"skip"`` values and exits
        non-zero.

    Raises
    ------
    RuntimeError
        If ``boto3`` is not installed (unauthenticated S3 head requires
        the AWS SDK; the AWS CLI alone won't work for arbitrary
        anonymous access against not-yet-public buckets).

    Notes
    -----
    Only ``s3://`` URLs are verified. ``https://`` URLs (e.g. the
    landing page) are out of scope — verify those manually with a
    browser before submission.
    """
    try:
        import boto3
        import botocore
    except ImportError as exc:
        raise RuntimeError(
            "boto3 not installed; install the distribution extra (uv sync --extra distribution) for verify_public"
        ) from exc

    payload = json.loads(croissant_path.read_text())
    urls = _iter_content_urls(payload)
    s3 = boto3.client(
        "s3",
        config=botocore.client.Config(signature_version=botocore.UNSIGNED),
    )
    statuses: dict[str, str] = {}
    for url in urls:
        if not url.startswith("s3://"):
            statuses[url] = "skip (non-s3 URL)"
            continue
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            statuses[url] = "missing-bucket"
            continue
        try:
            s3.head_object(Bucket=bucket, Key=key)
            statuses[url] = "ok"
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "unknown")
            statuses[url] = code
        except Exception as exc:  # pragma: no cover — surface unexpected
            statuses[url] = f"{type(exc).__name__}: {exc}"
    return statuses
