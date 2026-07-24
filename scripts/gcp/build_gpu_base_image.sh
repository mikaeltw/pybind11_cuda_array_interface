#!/usr/bin/env bash

set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required but not found in PATH." >&2
  exit 1
fi

: "${GCP_PROJECT:?Set GCP_PROJECT to the target project id}"
: "${GCP_ZONE:?Set GCP_ZONE to the zone to use}"

IMAGE_FAMILY="${IMAGE_FAMILY:-pybind11-cuda-array-interface-gpu-host}"
IMAGE_NAME="${IMAGE_NAME:-${IMAGE_FAMILY}-$(date +%Y%m%d-%H%M%S)}"
INSTANCE_NAME="${INSTANCE_NAME:-${IMAGE_NAME}-builder}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50}"
SOURCE_IMAGE_FAMILY="${SOURCE_IMAGE_FAMILY:-ubuntu-2204-lts}"
SOURCE_IMAGE_PROJECT="${SOURCE_IMAGE_PROJECT:-ubuntu-os-cloud}"
NETWORK="${NETWORK:-default}"
SUBNET="${SUBNET:-default}"
REGION="${REGION:-${GCP_ZONE%-*}}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-scripts/gcp/install_gpu_host.sh}"
PREEMPTIBLE="${PREEMPTIBLE:-true}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"
MAINTENANCE_POLICY="TERMINATE"

if [[ "${PREEMPTIBLE}" == "true" || "${PREEMPTIBLE}" == "1" ]]; then
  PROVISIONING_MODEL="SPOT"
  # Maintenance policy/automatic restart not supported for spot VMs.
  MAINTENANCE_POLICY=""
fi

echo "Using:"
echo "  Project:        ${GCP_PROJECT}"
echo "  Zone:           ${GCP_ZONE}"
echo "  Instance:       ${INSTANCE_NAME}"
echo "  Image name:     ${IMAGE_NAME}"
echo "  Machine type:   ${MACHINE_TYPE}"
echo "  GPU:            ${GPU_TYPE} x${GPU_COUNT}"
echo "  Boot disk (GB): ${BOOT_DISK_SIZE}"
echo "  Preemptible:    ${PREEMPTIBLE}"
echo "  Provisioning:   ${PROVISIONING_MODEL}"

wait_for_ssh() {
  local tries=30
  local sleep_seconds=10
  for ((i=1; i<=tries; i++)); do
    if gcloud compute ssh "${INSTANCE_NAME}" \
      --zone "${GCP_ZONE}" \
      --command "echo ready" \
      --quiet \
      -- \
      -o ConnectTimeout=5 \
      -o StrictHostKeyChecking=no \
      -o BatchMode=no >/dev/null 2>&1; then
      return 0
    fi
    echo "Waiting for SSH to become ready (${i}/${tries})..."
    sleep "${sleep_seconds}"
  done
  echo "SSH not reachable for ${INSTANCE_NAME} after $((tries * sleep_seconds))s." >&2
  gcloud compute instances delete "${INSTANCE_NAME}" --zone "${GCP_ZONE}" --quiet
  return 1
}

gcloud config set project "${GCP_PROJECT}" >/dev/null

gcloud compute instances create "${INSTANCE_NAME}" \
  --zone "${GCP_ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --accelerator "type=${GPU_TYPE},count=${GPU_COUNT}" \
  $(if [[ -n "${MAINTENANCE_POLICY}" ]]; then echo "--maintenance-policy ${MAINTENANCE_POLICY}"; fi) \
  --provisioning-model "${PROVISIONING_MODEL}" \
  --boot-disk-type=pd-ssd \
  --boot-disk-size "${BOOT_DISK_SIZE}"GB \
  --image-family "${SOURCE_IMAGE_FAMILY}" \
  --image-project "${SOURCE_IMAGE_PROJECT}" \
  --scopes cloud-platform \
  --network "${NETWORK}" \
  --subnet "${SUBNET}" \
  --tags pybind11-cuda-array-interface-gpu-builder \
  $(if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then echo "--instance-termination-action=DELETE"; fi)

wait_for_ssh

gcloud compute scp \
  --zone "${GCP_ZONE}" \
  "${INSTALL_SCRIPT}" \
  "${INSTANCE_NAME}:~/install_gpu_host.sh"

installer_env=()
[[ -n "${DOCKER_USER:-}" ]] && installer_env+=("DOCKER_USER=${DOCKER_USER}")
[[ -n "${INSTALL_NVIDIA_DRIVER:-}" ]] && installer_env+=("INSTALL_NVIDIA_DRIVER=${INSTALL_NVIDIA_DRIVER}")
[[ -n "${INSTALL_UV:-}" ]] && installer_env+=("INSTALL_UV=${INSTALL_UV}")
[[ -n "${LOCAL_SSD_DEVICE:-}" ]] && installer_env+=("LOCAL_SSD_DEVICE=${LOCAL_SSD_DEVICE}")

gcloud compute ssh "${INSTANCE_NAME}" \
  --zone "${GCP_ZONE}" \
  --command "chmod +x ~/install_gpu_host.sh && sudo ${installer_env[*]} ~/install_gpu_host.sh"

gcloud compute instances stop "${INSTANCE_NAME}" --zone "${GCP_ZONE}"

gcloud compute images create "${IMAGE_NAME}" \
  --source-disk "${INSTANCE_NAME}" \
  --source-disk-zone "${GCP_ZONE}" \
  --family "${IMAGE_FAMILY}"

gcloud compute instances delete "${INSTANCE_NAME}" --zone "${GCP_ZONE}" --quiet

echo "Image ${IMAGE_NAME} (family: ${IMAGE_FAMILY}) is ready in project ${GCP_PROJECT}."
