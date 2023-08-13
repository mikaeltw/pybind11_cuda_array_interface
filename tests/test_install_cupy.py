"""
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
"""

import sys
from unittest.mock import Mock, patch

import install_cupy
import pytest


def test_get_cuda_version():
    with patch("install_cupy.subprocess.run") as mock_run, patch("install_cupy.shutil.which") as mock_which:
        mock_which.return_value = "/path/to/nvcc"
        mock_run.return_value = Mock(stdout="nvcc: NVIDIA (R) Cuda compiler driver\n...\nrelease 11.2, V11.2.135")

        version = install_cupy.get_cuda_version()

        mock_which.assert_called_with("nvcc")
        mock_run.assert_called_with(["/path/to/nvcc", "--version"], capture_output=True, text=True, check=True)

        assert version == "112"


def test_install_cupy():
    with patch("install_cupy.get_cuda_version") as mock_version, patch("install_cupy.subprocess.run") as mock_run:
        mock_version.return_value = "112"

        install_cupy.install_cupy_matching_cuda_version()

        mock_version.assert_called_once()
        mock_run.assert_called_with([sys.executable, "-m", "pip", "install", "cupy-cuda112"], shell=False, check=True)


@pytest.mark.parametrize(
    "stdout, expected_version",
    [
        ("nvcc: NVIDIA (R) Cuda compiler driver\n...\nrelease 11.2, V11.2.135", "112"),
        ("nvcc: NVIDIA (R) Cuda compiler driver\n...\nrelease 10.1, V10.1.243", "101"),
    ],
)
def test_get_cuda_version_parametrized(stdout, expected_version):
    with patch("install_cupy.subprocess.run") as mock_run, patch("install_cupy.shutil.which") as mock_which:
        mock_which.return_value = "/path/to/nvcc"
        mock_run.return_value = Mock(stdout=stdout)

        version = install_cupy.get_cuda_version()

        assert version == expected_version


if __name__ == "__main__":
    install_cupy.install_cupy_matching_cuda_version()
