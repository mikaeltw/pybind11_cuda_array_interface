"""
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
"""

import fnmatch
import os
import subprocess
import sys
from typing import List


class Linter:
    def __init__(self, directory: str):
        self.directory = directory
        self.ignore_list = [
            ".git",
            "__pycache__",
            "thirdparty",
            "build",
            ".env",
            ".github",
            "dist",
            ".eggs",
            "*.egg",
            "_skbuild",
            "venv",
        ]

    def get_python_files(self) -> List[str]:
        python_files = []

        for root, dirs, files in os.walk(self.directory):
            # ignore directories in the ignore_list
            dirs[:] = [d for d in dirs if d not in self.ignore_list]

            for filename in fnmatch.filter(files, "*.py"):
                python_files.append(os.path.join(root, filename))

        return python_files

    def run_linters(self) -> None:
        python_files = self.get_python_files()

        commands = [
            [sys.executable, "-m", "black", ".", "--check"],
            [sys.executable, "-m", "isort", ".", "--check-only"],
            [sys.executable, "-m", "mypy", "--install-types", "--non-interactive"],
            [sys.executable, "-m", "mypy", "--no-incremental"],
            [sys.executable, "-m", "flake8"],
            [sys.executable, "-m", "pylint", *python_files],
            [sys.executable, "-m", "bandit", "-c", "pyproject.toml", "-r", "."],
        ]
        return_codes = []
        for command in commands:
            print("".join(("\nRunning ", command[2], "\n")))
            process = subprocess.run(command, shell=False, check=False, cwd=self.directory)
            return_codes.append(process.returncode)

        if all(v == 0 for v in return_codes):
            print("\nLinting succeeded 🎉\n")
        else:
            print("\nLinting failed 🔗‍💥\n")
            sys.exit(1)


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # define parent directory
    linter = Linter(parent_dir)
    linter.run_linters()
