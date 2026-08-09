#!/usr/bin/env bash

set -Eeuo pipefail

readonly SOURCE_DIR="/workspace"
readonly BUILD_DIR="$(mktemp -d /tmp/gpu-tests-build.XXXXXX)"
readonly CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75}"

cleanup() {
  rm -rf -- "${BUILD_DIR}"
}
trap cleanup EXIT

cmake \
  -S "${SOURCE_DIR}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DBUILD_GTESTS=ON \
  -DBUILD_PYTESTS=ON \
  -DCOPY_TEST_ARTIFACTS_TO_SOURCE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

cmake --build "${BUILD_DIR}" --parallel

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${BUILD_DIR}/tests" \
  python -m pytest -p no:cacheprovider "${SOURCE_DIR}/tests/pytest"

"${BUILD_DIR}/tests/run_gtest_cai"
