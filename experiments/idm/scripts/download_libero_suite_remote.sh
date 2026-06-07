#!/usr/bin/env bash
set -euo pipefail

# Remote-only script. Run this on the EC2 instance, not on a laptop.
# Example:
#   DATASET=libero_10 bash experiments/idm/scripts/download_libero_suite_remote.sh

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
LIBERO_REPO="${LIBERO_REPO:-${ROBOTICS_ROOT}/repos/LIBERO}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${ROBOTICS_ROOT}/data/libero}"
PYTHON="${PYTHON:-python3}"
DATASET="${DATASET:-libero_spatial}"

mkdir -p "${ROBOTICS_ROOT}/repos" "${LIBERO_DATA_ROOT}"

if [[ ! -d "${LIBERO_REPO}/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_REPO}"
fi

cd "${LIBERO_REPO}"
"${PYTHON}" benchmark_scripts/download_libero_datasets.py --datasets "${DATASET}"

mkdir -p "${LIBERO_DATA_ROOT}"
if [[ -d "${LIBERO_REPO}/datasets/${DATASET}" ]]; then
  rsync -a "${LIBERO_REPO}/datasets/${DATASET}/" "${LIBERO_DATA_ROOT}/${DATASET}/"
fi

find "${LIBERO_DATA_ROOT}/${DATASET}" -maxdepth 1 -type f \( -name '*.hdf5' -o -name '*.h5' \) | sort
