import os
import toml
import re
import sys


from skbuild import setup
from setuptools import find_packages
from setuptools_scm import get_version


def minimum_requirements_of_non_python_dependencies():
    return ['cuda>=10.1',
            'cxx==14',]


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


def get_version_of_package_as_dependency_str():
    version = get_version()
    return [''.join(('package==', version))]


def get_dependencies_as_cmake_args():
    # Load the pyproject.toml file
    pyproject = toml.load('pyproject.toml')

    # Get the build system requirements
    build_requires = pyproject['build-system']['requires']
    python_requires = [''.join(('python', pyproject['project']['requires-python']))]
    non_python_requires = minimum_requirements_of_non_python_dependencies()
    version_requires = get_version_of_package_as_dependency_str()

    package_list = build_requires + python_requires + non_python_requires + version_requires

    separators = ['>=', '==']
    separator_pattern = '|'.join(re.escape(separator) for separator in separators)
    versions = {key: value for key, value in (re.split(separator_pattern, s) for s in package_list if re.search(separator_pattern, s))}

    cmake_dependency_args = []
    for pkg, version in versions.items():
        cmake_dependency_args.append(f'-D{pkg.upper()}_MVERSION={version}')

    return cmake_dependency_args


def get_testbuild_args_as_cmake_args():
    return ['-DDOWNLOAD_GTEST=ON',
            '-DBUILD_GTESTS=ON',
            '-DBUILD_PYTESTS=ON']


def get_linux_cmake_args():
    return ['-DCMAKE_INSTALL_LIBDIR=lib',]


def get_windows_cmake_args():
    ['-GNinja',
     '-Dgtest_force_shared_crt=ON',]


def get_macosx_cmake_args():
    return ['-DCMAKE_INSTALL_LIBDIR=lib',]


def get_cmake_args(dev=True):
    cmake_args = []
    if dev:
        cmake_args += get_testbuild_args_as_cmake_args()
    LINUX = sys.platform.startswith("linux")
    MACOS = sys.platform.startswith("darwin")
    WIN = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")
    if LINUX:
        cmake_args += get_linux_cmake_args()
    elif MACOS:
        cmake_args += get_macosx_cmake_args()
    elif WIN:
        cmake_args += get_windows_cmake_args()
    else:
        assert("Unknown platform.")
    return cmake_args


setup(
    packages=find_packages(),
    platforms=["Linux", "Windows", "MacOSX"],
    #cmake_install_dir=get_cmake_build_dir(),
    #cmake_source_dir=get_current_directory(),
    cmake_args=get_cmake_args()
)
