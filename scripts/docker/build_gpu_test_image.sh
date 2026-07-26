#!/usr/bin/env bash

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the GPU test image." >&2
  exit 1
fi

# Default to a regional Artifact Registry repo close to the GPU VMs.
GCP_PROJECT="${GCP_PROJECT:-gpu-test-runners}"
GCP_ARTIFACT_REGION="${GCP_ARTIFACT_REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-pybind11-cuda-array-interface}"
PACKAGE="${PACKAGE:-pybind11-cuda-array-interface-gpu-tests}"
IMAGE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"

IMAGE_REF="${GCP_ARTIFACT_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPOSITORY}/${PACKAGE}:${IMAGE_VERSION}"

CUDA_IMAGE_DEVEL="${CUDA_IMAGE_DEVEL:-nvidia/cuda:13.0.3-devel-ubuntu24.04}"
CUDA_IMAGE_RUNTIME="${CUDA_IMAGE_RUNTIME:-nvidia/cuda:13.0.3-base-ubuntu24.04}"

echo "Building GPU test image ${IMAGE_REF} (devel=${CUDA_IMAGE_DEVEL}, runtime=${CUDA_IMAGE_RUNTIME})"

docker build \
  -f docker/gpu-tests.Dockerfile \
  --no-cache \
  --build-arg "CUDA_IMAGE_DEVEL=${CUDA_IMAGE_DEVEL}" \
  --build-arg "CUDA_IMAGE_RUNTIME=${CUDA_IMAGE_RUNTIME}" \
  -t "${IMAGE_REF}" \
  .

cat <<EOF
Built image: ${IMAGE_REF}
To run GPU tests locally:
GCP_ARTIFACT_REGION="${GCP_ARTIFACT_REGION}" GCP_PROJECT="${GCP_PROJECT}" REPOSITORY="${REPOSITORY}" PACKAGE="${PACKAGE}" IMAGE_VERSION="${IMAGE_VERSION}" ./scripts/docker/run_gpu_tests.sh

If you wish to push it:
docker push "${IMAGE_REF}"
EOF
