#!/usr/bin/env bash

# Cross-platform packaging of dynamically linked libraries
# for wheels.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: repair_wheel.sh [wheel_dir] [output_dir]

Examples:
  repair_wheel.sh dist
  repair_wheel.sh dist repaired
  repair_wheel.sh

Defaults:
  wheel_dir = dist
  output_dir = wheel_dir
USAGE
}

wheel_dir=${1:-dist}
output_dir=${2:-${wheel_dir}}

if [ "${wheel_dir}" = "-h" ] || [ "${wheel_dir}" = "--help" ]; then
  usage
  exit 0
fi

shopt -s nullglob
wheels=("${wheel_dir}"/*.whl)
shopt -u nullglob

if [ ${#wheels[@]} -eq 0 ]; then
  echo "no wheels found in ${wheel_dir}" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin)
    if ! command -v delocate-wheel >/dev/null 2>&1; then
      echo "delocate-wheel not found; install with: python -m pip install delocate" >&2
      exit 1
    fi
    for wheel in "${wheels[@]}"; do
      delocate-wheel -w "${output_dir}" "${wheel}"
    done
    ;;
  Linux)
    if ! command -v auditwheel >/dev/null 2>&1; then
      echo "auditwheel not found; install with: python -m pip install auditwheel" >&2
      exit 1
    fi
    for wheel in "${wheels[@]}"; do
      auditwheel repair "${wheel}" -w "${output_dir}"
    done
    ;;
  MINGW*|MSYS*|CYGWIN*)
    if ! command -v delvewheel >/dev/null 2>&1; then
      echo "delvewheel not found; install with: python -m pip install delvewheel" >&2
      exit 1
    fi
    for wheel in "${wheels[@]}"; do
      delvewheel repair "${wheel}" -w "${output_dir}"
    done
    ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac
