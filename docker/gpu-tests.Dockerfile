ARG CUDA_IMAGE_DEVEL=nvidia/cuda:13.0.3-devel-ubuntu24.04

FROM ${CUDA_IMAGE_DEVEL}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gpu-tests-venv \
    PATH="/opt/gpu-tests-venv/bin:${PATH}" \
    CUDA_PATH=/usr/local/cuda/targets/x86_64-linux \
    PYBIND11_CUDA_ARRAY_INTERFACE_DEVICE=gpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ninja-build \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      "cupy-cuda13x" \
      "numpy>=1.23" \
      "pybind11>=3,<4" \
      "pytest>=7"

# The repository is mounted at runtime. Keeping source and compiled test
# artifacts out of this image ensures CI always builds the checked-out commit.
WORKDIR /workspace
CMD ["bash"]
