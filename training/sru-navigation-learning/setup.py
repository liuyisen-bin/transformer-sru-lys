#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="2.3.4",
    packages=find_packages(),
    author="Fan Yang, ETH Zurich",
    maintainer="Fan Yang",
    maintainer_email="fanyang1@ethz.ch",
    license="BSD-3",
    description="SRU Navigation Learning - RL training framework with Spatially-Enhanced Recurrent Units for visual navigation",
    python_requires=">=3.11,<3.12",
    install_requires=[
        "torch>=2.7,<2.8",
        "torchvision>=0.22,<0.23",
        "numpy>=1.26,<2",
        "GitPython",
        "onnx",
        "tensorboard",
    ],
)
