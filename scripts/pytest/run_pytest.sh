#!/usr/bin/env bash
#
# Requires: pip install pytest pytest-xdist
#
set -euo pipefail
jobs="$(nproc)"
dir="$(dirname "$0")"

# Reduce jobs to at max 8, as pytest jobs setup is actually costly
[ "$jobs" -le 8 ] || jobs=8

# Get to top dir as pytest does not discover pyproject.toml otherwise
cd "$dir"/../..

set -x
exec pytest -n "$jobs" "$@"
