import os
import re
import shutil
import sys

import toml
from setuptools_scm import get_version
from skbuild import setup


def get_env_variable_bool_state(var_name):
    value = os.getenv(var_name)
    if value is not None and value.lower() in ("1", "true", "yes", "on"):
        return "ON"
    else:
        return "OFF"


def minimum_requirements_of_non_python_dependencies():
    return [
        "cuda>=10.1",
        "cxx==14",
    ]


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
    return ["".join(("package==", version))]


def get_dependencies_as_cmake_args():
    # Load the pyproject.toml file
    pyproject = toml.load("pyproject.toml")

    # Get the build system requirements
    build_requires = pyproject["build-system"]["requires"]
    python_requires = ["".join(("python", pyproject["project"]["requires-python"]))]
    non_python_requires = minimum_requirements_of_non_python_dependencies()
    version_requires = get_version_of_package_as_dependency_str()

    package_list = build_requires + python_requires + non_python_requires + version_requires

    separators = [">=", "=="]
    separator_pattern = "|".join(re.escape(separator) for separator in separators)
    versions = {
        key: value
        for key, value in (re.split(separator_pattern, s) for s in package_list if re.search(separator_pattern, s))
    }

    cmake_dependency_args = []
    for pkg, version in versions.items():
        cmake_dependency_args.append(f"-D{pkg.upper()}_MVERSION={version}")

    return cmake_dependency_args


def get_testbuild_args_as_cmake_args():
    return [
        "".join(("-DBUILD_GTESTS=", get_env_variable_bool_state("BUILD_GTESTS"))),
        "".join(("-DBUILD_PYTESTS=", get_env_variable_bool_state("BUILD_PYTESTS"))),
    ]


def get_linux_cmake_args():
    return [
        "-DCMAKE_INSTALL_LIBDIR=lib",
    ]


def get_windows_cmake_args():
    [
        "-GNinja",
        "-Dgtest_force_shared_crt=ON",
    ]


def get_macosx_cmake_args():
    return [
        "-DCMAKE_INSTALL_LIBDIR=lib",
    ]


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
        assert "Unknown platform."
    cmake_args += get_dependencies_as_cmake_args()
    return cmake_args


if __name__ == "__main__":
    # Make sure that we start with a clean build every time.
    build_path = os.path.join(get_current_directory(abs_path=True), "_skbuild")
    shutil.rmtree(build_path, ignore_errors=True)

    setup(cmake_args=get_cmake_args())
