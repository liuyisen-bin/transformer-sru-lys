# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'engineai_rl_lab' python package."""

import os
from pathlib import Path

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "onnx",
    "psutil",
    "numpy>=1.26,<2",
    "scipy",
    "trimesh",
    "matplotlib",
]

setup(
    name="engineai_rl_lab",
    packages=find_packages(),
    package_data={
        "engineai_rl_lab": [
            str(path.relative_to(Path(EXTENSION_PATH) / "engineai_rl_lab"))
            for path in (Path(EXTENSION_PATH) / "engineai_rl_lab/tasks/navigation/assets/data").rglob("*")
            if path.is_file()
        ],
    },
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.11,<3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
