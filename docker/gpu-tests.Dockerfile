ARG CUDA_IMAGE_DEVEL=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04
ARG CUDA_IMAGE_RUNTIME=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04


FROM ${CUDA_IMAGE_DEVEL} AS build

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gpu-tests-venv \
    PATH="/opt/gpu-tests-venv/bin:${PATH}"

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

WORKDIR /opt/pybind11_cuda_array_interface
COPY . .

# The Cirun runner uses an NVIDIA T4 (compute capability 7.5). Build both test
# frontends here so the GPU job only has to execute them.
RUN cmake \
      -S . \
      -B /opt/gpu-tests-build \
      -G Ninja \
      -DBUILD_GTESTS=ON \
      -DBUILD_PYTESTS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" && \
    cmake --build /opt/gpu-tests-build --parallel


FROM ${CUDA_IMAGE_RUNTIME} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gpu-tests-venv \
    PATH="/opt/gpu-tests-venv/bin:${PATH}" \
    PYTHONPATH=/opt/pybind11_cuda_array_interface/tests/pytest \
    PYBIND11_CUDA_ARRAY_INTERFACE_DEVICE=gpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libgomp1 \
      libpython3.12t64 \
      python3 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/gpu-tests-venv /opt/gpu-tests-venv
COPY --from=build /opt/pybind11_cuda_array_interface /opt/pybind11_cuda_array_interface

WORKDIR /opt/pybind11_cuda_array_interface
CMD ["python", "-m", "pytest", "/opt/pybind11_cuda_array_interface/tests/pytest"]
