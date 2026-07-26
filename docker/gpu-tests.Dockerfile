ARG CUDA_IMAGE_DEVEL=nvidia/cuda:13.0.3-devel-ubuntu24.04
ARG CUDA_IMAGE_RUNTIME=nvidia/cuda:13.0.3-base-ubuntu24.04


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
    cmake --build /opt/gpu-tests-build --parallel && \
    strip tests/pytest/pycai.so tests/gtest/run_gtest_cai && \
    mkdir /opt/nvrtc-libs && \
    cp -a \
      /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so* \
      /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so* \
      /opt/nvrtc-libs/ && \
    python -m pip uninstall --yes pybind11 pip


FROM ${CUDA_IMAGE_RUNTIME} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gpu-tests-venv \
    PATH="/opt/gpu-tests-venv/bin:${PATH}" \
    CUDA_PATH=/usr/local/cuda/targets/x86_64-linux \
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
# The CUDA base image supplies cudart. CuPy additionally needs NVRTC and CUDA
# headers for its elementwise/reduction kernels; copying just those files is
# much smaller than the CUDA runtime/development images and their unused math
# libraries.
COPY --from=build /opt/nvrtc-libs/ /usr/local/cuda/targets/x86_64-linux/lib/
COPY --from=build /usr/local/cuda/targets/x86_64-linux/include /usr/local/cuda/targets/x86_64-linux/include

WORKDIR /opt/pybind11_cuda_array_interface
CMD ["python", "-m", "pytest", "/opt/pybind11_cuda_array_interface/tests/pytest"]
