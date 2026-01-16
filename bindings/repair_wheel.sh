#!/usr/bin/env bash

# Cross-platform packaging of dynamically linked libraries
# for wheels.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: repair_wheel.sh [wheel_glob] [output_dir]

Examples:
  repair_wheel.sh "dist/*.whl" dist
  repair_wheel.sh

Defaults:
  wheel_glob = dist/*.whl
  output_dir = dist
USAGE
}

wheel_glob=${1:-dist/*.whl}
output_dir=${2:-dist}

if [ "${wheel_glob}" = "-h" ] || [ "${wheel_glob}" = "--help" ]; then
  usage
  exit 0
fi

case "$(uname -s)" in
  Darwin)
    if ! command -v delocate-wheel >/dev/null 2>&1; then
      echo "delocate-wheel not found; install with: python -m pip install delocate" >&2
      exit 1
    fi
    delocate-wheel -w "${output_dir}" ${wheel_glob}
    ;;
  Linux)
    if ! command -v auditwheel >/dev/null 2>&1; then
      echo "auditwheel not found; install with: python -m pip install auditwheel" >&2
      exit 1
    fi
    auditwheel repair ${wheel_glob} -w "${output_dir}"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    if ! command -v delvewheel >/dev/null 2>&1; then
      echo "delvewheel not found; install with: python -m pip install delvewheel" >&2
      exit 1
    fi
    delvewheel repair ${wheel_glob} -w "${output_dir}"
    ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac
