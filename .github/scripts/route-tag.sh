#!/usr/bin/env bash
# Route a release tag to its workspace package and version.
#
# Usage: route-tag.sh <tag>
# Prints `package=<name>` and `version=<version>` (GitHub Actions output format)
# to stdout; a human-readable line goes to stderr. Exits 1 with an ::error::
# annotation on an unrecognized tag.
#
# Each arm is anchored on `v[0-9]*`, and the bare-v umbrella arm stays LAST, so a
# `viscy-…` tag or a subpackage tag missing its `v` (e.g. viscy-data-0.5.0) can't
# fall through to the umbrella with a garbage version. Adding a package = add one
# arm. The prefixes mirror each package's `pattern-prefix` in its pyproject.toml.
set -euo pipefail

tag="${1:?usage: route-tag.sh <tag>}"

case "$tag" in
  # libraries (packages/)
  viscy-data-v[0-9]*)       package=viscy-data;       version=${tag#viscy-data-v} ;;
  viscy-models-v[0-9]*)     package=viscy-models;     version=${tag#viscy-models-v} ;;
  viscy-transforms-v[0-9]*) package=viscy-transforms; version=${tag#viscy-transforms-v} ;;
  viscy-utils-v[0-9]*)      package=viscy-utils;      version=${tag#viscy-utils-v} ;;
  # applications (applications/)
  airtable-utils-v[0-9]*)   package=airtable-utils;   version=${tag#airtable-utils-v} ;;
  cytoland-v[0-9]*)         package=cytoland;         version=${tag#cytoland-v} ;;
  dynacell-v[0-9]*)         package=dynacell;         version=${tag#dynacell-v} ;;
  dynaclr-v[0-9]*)          package=dynaclr;          version=${tag#dynaclr-v} ;;
  viscy-qc-v[0-9]*)         package=viscy-qc;         version=${tag#viscy-qc-v} ;;
  # umbrella (must stay last)
  v[0-9]*)                  package=viscy;            version=${tag#v} ;;
  *) echo "::error::Unrecognized release tag '$tag' (expected '<pkg>-vX.Y.Z' or 'vX.Y.Z')" >&2; exit 1 ;;
esac

echo "Routed $tag -> package=$package version=$version" >&2
printf 'package=%s\nversion=%s\n' "$package" "$version"
