import re
import subprocess
import sys


def get_cuda_version() -> str:
    version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
    version = re.split("release ", version.stdout)[1]  # get major and minor version
    version = re.split(r"\.| |,", version)[0:2]
    return "".join(version)


def install_cupy() -> None:
    cuda_version = get_cuda_version()
    cupy_package = "".join(("cupy-cuda", cuda_version))
    subprocess.run([sys.executable, "-m", "pip", "install", cupy_package], check=True)


if __name__ == "__main__":
    install_cupy()
