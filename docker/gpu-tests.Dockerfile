ARG CUDA_IMAGE_DEVEL=nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04
ARG CUDA_IMAGE_RUNTIME=nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04


FROM ${CUDA_IMAGE_DEVEL} AS base-build

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/pybind11_cuda_array_interface/.venv \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      pkg-config \
      python3 \
      python3-venv \
      python3-pip \
      python3-dev \
      make \
      jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

FROM base-build AS deps

COPY pyproject.toml ./
# COPY pyproject.toml uv.lock README.md LICENSE Makefile ./
# COPY src ./src

# RUN make setup && make sync && \
#     uv python install 3.13 && \
#     uv python install 3.14 && \
#     uv cache prune --ci

FROM ${CUDA_IMAGE_RUNTIME} AS runtime

ENV UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/pybind11_cuda_array_interface/.venv \
    PATH="/root/.local/bin:${PATH}"

# Minimal runtime packages to run the tests/venv (single layer, aggressive cleanup)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      make \
      python3 \
      python3-venv \
      python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man /usr/share/info /tmp/* /var/tmp/*

# COPY --from=deps /root/.local /root/.local
# COPY --from=deps /opt/pybind11_cuda_array_interface/.venv /opt/pybind11_cuda_array_interface/.venv

ENTRYPOINT ["/bin/bash"]
