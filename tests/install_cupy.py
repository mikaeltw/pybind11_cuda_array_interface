import subprocess
import sys
import re
import os

def get_cuda_version():
    version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    version = version.stdout.split()[-1]  # get the version string
    version = re.split('_|\.', version)[1:3] # get major and minor version
    return "".join(version)

def install_cupy():
    cuda_version = get_cuda_version()
    cupy_package = "".join(("cupy-cuda", cuda_version))
    subprocess.run([sys.executable, "-m", "pip", "install", cupy_package])

if __name__ == "__main__":
    print(os.environ)
    install_cupy()