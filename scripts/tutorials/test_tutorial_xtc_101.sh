#!/usr/bin/env bash
#
# Test xtc_101 tutorial
#
set -euo pipefail

dir="$(dirname "$0")"

MARIMO="$dir"/../../docs/tutorials/xtc_101.py
set -x
python "$MARIMO"
