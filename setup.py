import os



from skbuild import setup
from setuptools import find_packages


with open("README.md", "r") as f:
    long_description = f.read()


def get_current_directory(abs_path=False):
    if abs_path:
        return os.path.dirname(os.path.realpath(__file__))
    else:
        return "."

def get_cmake_build_dir(abs_path=False, cmake_build_dir="build"):
    if abs_path:
        DIRECTORY = get_current_directory()
        path = os.path.join(DIRECTORY, cmake_build_dir)
        return path
    else:
        return cmake_build_dir


def get_cmake_args(dev=True):
    if dev:
        return ['-DDOWNLOAD_GTEST=ON',
                '-DBUILD_GTESTS=ON',
                '-DBUILD_PYTESTS=ON']
    else:
        return ['']



# if not os.path.exists(get_cmake_build_dir()):
#     os.makedirs(get_cmake_build_dir())

setup(
    name='pybind11_cuda_array_interface',
    version='0.0.1',
    packages=find_packages(),
    author="Mikael Twengström",
    author_email="m.twengstrom@gmail.com",
    description="pybind11 headers supporting the __cuda_array_interface__",
    long_description=long_description,
    license="BSD 3-Clause",
    platforms=["Linux", "Windows", "MacOSX"],
    long_description_content_type='text/markdown',
    url="https://github.com/mikaeltw/pybind11_cuda_array_interface",
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD 3-Clause",
    ],
    install_requires=[''],
    python_requires=">=3.7",
    #cmake_install_dir=get_cmake_build_dir(),
    #cmake_source_dir=get_current_directory(),
    cmake_args=get_cmake_args()
)
