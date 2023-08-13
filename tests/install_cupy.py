"""
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
"""

import re
import shutil
import subprocess
import sys


def get_cuda_version() -> str:
    nvcc = shutil.which("nvcc")
    version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True)
    version = re.split("release ", version.stdout)[1]  # get major and minor version
    version = re.split(r"\.| |,", version)[0:2]
    return "".join(version)


def install_cupy_matching_cuda_version() -> None:
    cuda_version = get_cuda_version()
    cupy_package = "".join(("cupy-cuda", cuda_version))
    subprocess.run([sys.executable, "-m", "pip", "install", cupy_package], shell=False, check=True)


if __name__ == "__main__":
    install_cupy_matching_cuda_version()
