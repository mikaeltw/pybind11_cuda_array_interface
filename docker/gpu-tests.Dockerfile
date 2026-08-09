ARG CUDA_IMAGE_BASE=nvidia/cuda:13.0.3-base-ubuntu24.04

FROM ${CUDA_IMAGE_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gpu-tests-venv \
    PATH="/opt/gpu-tests-venv/bin:${PATH}" \
    CUDA_PATH=/usr/local/cuda/targets/x86_64-linux \
    PYBIND11_CUDA_ARRAY_INTERFACE_DEVICE=gpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
      cuda-cudart-dev-13-0 \
      cuda-nvcc-13-0 \
      cuda-nvrtc-13-0 \
      cuda-nvrtc-dev-13-0 \
      g++ \
      ninja-build \
      python3 \
      python3-dev \
      python3-venv && \
    rm -f \
      /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc_static.a \
      /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc_static.alt.a \
      /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins_static.a \
      /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins_static.alt.a && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      "cupy-cuda13x" \
      "numpy>=1.23" \
      "pybind11>=3,<4" \
      "pytest>=7" && \
    python -m pip uninstall --yes pip

# The base image plus the CUDA component packages above is considerably smaller
# than the CUDA devel image while retaining nvcc, cudart, and NVRTC. The
# repository is mounted at runtime so CI always builds the checked-out commit.
WORKDIR /workspace
CMD ["bash"]
