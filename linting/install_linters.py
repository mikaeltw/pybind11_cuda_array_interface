import os
import subprocess

import toml


def install_optional_dependencies():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pyproject.toml")
    with open(path) as f:
        pyproject = toml.load(f)

    optional_dependencies = pyproject["project"]["optional-dependencies"]

    for deps in optional_dependencies.values():
        subprocess.run([sys.executable, "-m", "pip", "install"] + deps, check=True)


if __name__ == "__main__":
    install_optional_dependencies()
