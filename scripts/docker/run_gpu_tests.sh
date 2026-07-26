#!/usr/bin/env bash

set -euo pipefail

GCP_ARTIFACT_REGION="${GCP_ARTIFACT_REGION:-us-central1}"
GCP_PROJECT="${GCP_PROJECT:-gpu-test-runners}"
REPOSITORY="${REPOSITORY:-pybind11-cuda-array-interface}"
PACKAGE="${PACKAGE:-pybind11-cuda-array-interface-gpu-tests}"
IMAGE_VERSION="${IMAGE_VERSION:-latest}"

IMAGE_REF="${GCP_ARTIFACT_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPOSITORY}/${PACKAGE}:${IMAGE_VERSION}"

DOCKER_ENTRYPOINT=()
if [[ $# -eq 0 ]]; then
  DOCKER_ENTRYPOINT=(--entrypoint /bin/bash)
  COMMAND=(-lc "python -m pytest /opt/pybind11_cuda_array_interface/tests/pytest && /opt/pybind11_cuda_array_interface/tests/gtest/run_gtest_cai")
else
  COMMAND=("$@")
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to run GPU tests inside a container." >&2
  exit 1
fi

gcloud auth configure-docker ${GCP_ARTIFACT_REGION}-docker.pkg.dev --quiet

docker pull "${IMAGE_REF}" || echo "Pull failed (probably have not pushed yet). Using local image."

docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  -e PYBIND11_CUDA_ARRAY_INTERFACE_DEVICE=gpu \
  "${DOCKER_ENTRYPOINT[@]}" \
  "${IMAGE_REF}" \
  "${COMMAND[@]}"
