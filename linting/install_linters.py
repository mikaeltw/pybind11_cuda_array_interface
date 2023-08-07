"""
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
"""

import os
import subprocess
import sys

import toml


def install_optional_dependencies() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pyproject.toml")
    with open(path) as f:
        pyproject = toml.load(f)

    optional_dependencies = pyproject["project"]["optional-dependencies"]

    for deps in optional_dependencies.values():
        subprocess.run([sys.executable, "-m", "pip", "install"] + deps, check=True)


if __name__ == "__main__":
    install_optional_dependencies()
