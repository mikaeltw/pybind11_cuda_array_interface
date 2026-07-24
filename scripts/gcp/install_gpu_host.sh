#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must be run as root. Try 'sudo $0'." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

INSTALL_NVIDIA_DRIVER="${INSTALL_NVIDIA_DRIVER:-1}"
INSTALL_UV="${INSTALL_UV:-0}"
DOCKER_USER="${DOCKER_USER:-ci-gpu-runner}"
LOCAL_SSD_DEVICE="${LOCAL_SSD_DEVICE:-}"

apt-get update
apt-get install -y --no-install-recommends \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  jq

# Install Docker CE
install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
fi
chmod a+r /etc/apt/keyrings/docker.gpg

DOCKER_RELEASE=$(. /etc/os-release && echo "${VERSION_CODENAME}")
cat >/etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${DOCKER_RELEASE} stable
EOF

apt-get update
apt-get install -y --no-install-recommends \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
systemctl enable --now docker
# Create/choose a non-root account to grant docker access. Allow override via DOCKER_USER.
if [[ -n "${DOCKER_USER:-}" ]] && ! id "${DOCKER_USER}" &>/dev/null; then
  useradd -m -s /bin/bash "${DOCKER_USER}"
  echo "Added DOCKER_USER: ${DOCKER_USER}"
fi
# Pick a non-root account to grant docker access. Allow override via DOCKER_USER.
pick_default_user() {
  # Preferred order: explicit override, sudo caller, uid 1000, first uid>=1000
  if [[ -n "${DOCKER_USER:-}" ]] && id "${DOCKER_USER}" &>/dev/null; then
    echo "${DOCKER_USER}"
    return
  fi
  if [[ -n "${SUDO_USER:-}" ]] && id "${SUDO_USER}" &>/dev/null; then
    echo "${SUDO_USER}"
    return
  fi
  if user_1000="$(getent passwd 1000 | cut -d: -f1 2>/dev/null)" && [[ -n "${user_1000}" ]]; then
    echo "${user_1000}"
    return
  fi
  awk -F: '$3>=1000 && $1!="nobody"{print $1; exit}' /etc/passwd
}
default_user="$(pick_default_user || true)"
if [[ -n "${default_user}" ]] && id "${default_user}" &>/dev/null; then
  usermod -aG docker "${default_user}"
fi

# Configure Docker to use a local SSD for its data-root when present.
cat >/usr/local/sbin/setup-docker-local-ssd.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

MOUNT_POINT="${MOUNT_POINT:-/mnt/local-ssd}"
DATA_ROOT="${DOCKER_DATA_ROOT:-${MOUNT_POINT}/docker}"

pick_ephemeral() {
  # Prefer NVMe disks that are not the boot disk. Fallback: any device whose
  # model contains "EphemeralDisk".
  while read -r name model type mount; do
    # Skip loop/partitions/mounted root.
    if [[ "${type}" != "disk" ]]; then
      continue
    fi
    if [[ "${mount}" == "/" ]]; then
      continue
    fi
    # NVMe local SSDs typically show up as nvme0n1 with model "nvme_card".
    if [[ "${name}" == nvme* ]]; then
      echo "/dev/${name}"
      return
    fi
    if [[ "${model}" == *"EphemeralDisk"* ]]; then
      echo "/dev/${name}"
      return
    fi
  done < <(lsblk -ndo NAME,MODEL,TYPE,MOUNTPOINT)
}

device="${LOCAL_SSD_DEVICE:-}"
if [[ -z "${device}" ]]; then
  device="$(pick_ephemeral || true)"
fi

# If no local SSD is present, exit silently.
if [[ -z "${device}" || ! -b "${device}" ]]; then
  exit 0
fi

# Format if needed and mount.
if ! blkid "${device}" >/dev/null 2>&1; then
  mkfs.ext4 -F "${device}"
fi
mkdir -p "${MOUNT_POINT}"
if ! mountpoint -q "${MOUNT_POINT}"; then
  mount "${device}" "${MOUNT_POINT}"
fi

mkdir -p "${DATA_ROOT}"
daemon="/etc/docker/daemon.json"
if [[ ! -s "${daemon}" ]]; then
  echo '{}' > "${daemon}"
fi
tmp="$(mktemp)"
jq --arg data_root "${DATA_ROOT}" '.["data-root"]=$data_root' "${daemon}" >"${tmp}"
mv "${tmp}" "${daemon}"
EOF
chmod +x /usr/local/sbin/setup-docker-local-ssd.sh

cat >/etc/systemd/system/setup-docker-local-ssd.service <<'EOF'
[Unit]
Description=Configure Docker data root on local SSD if present
After=local-fs.target
Before=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/setup-docker-local-ssd.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl enable setup-docker-local-ssd.service

# Install NVIDIA driver if requested
if [[ "${INSTALL_NVIDIA_DRIVER}" == "1" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends nvidia-driver-580
fi

# Install NVIDIA container toolkit
install -m 0755 -d /usr/share/keyrings
distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Install uv (available to default user)
if [[ "${INSTALL_UV}" == "1" ]]; then
  target_user="${SUDO_USER:-ubuntu}"
  if id "${target_user}" &>/dev/null; then
    sudo -u "${target_user}" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  fi
fi

apt-get clean
rm -rf /var/lib/apt/lists/*

echo "GPU host setup complete."
