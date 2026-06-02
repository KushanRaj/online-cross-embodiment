#!/usr/bin/env bash
set -euo pipefail

# Remote-only script. Run this on the EC2 instance, not on a laptop.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
LIBERO_REPO="${LIBERO_REPO:-${ROBOTICS_ROOT}/repos/LIBERO}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${ROBOTICS_ROOT}/data/libero}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${ROBOTICS_ROOT}/repos" "${LIBERO_DATA_ROOT}"

if [[ ! -d "${LIBERO_REPO}/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_REPO}"
fi

cd "${LIBERO_REPO}"

# Official LIBERO dataset entrypoint:
#   python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
#
# The script writes under ./datasets by default. We move/sync the resulting
# libero_spatial folder into the common robotics data root used by our manifests.
"${PYTHON}" benchmark_scripts/download_libero_datasets.py --datasets libero_spatial

mkdir -p "${LIBERO_DATA_ROOT}"
if [[ -d "${LIBERO_REPO}/datasets/libero_spatial" ]]; then
  rsync -a "${LIBERO_REPO}/datasets/libero_spatial/" "${LIBERO_DATA_ROOT}/libero_spatial/"
fi

find "${LIBERO_DATA_ROOT}/libero_spatial" -maxdepth 1 -type f \( -name '*.hdf5' -o -name '*.h5' \) | sort
