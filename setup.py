# -*- coding: utf-8 -*-

import os
import importlib

from setuptools import setup, find_packages

base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "meshtools", "__about__.py"), "rb") as fh:
    exec(fh.read(), about)


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="meshtools",
    packages=find_packages(),
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    install_requires=["numpy", "meshio", "pygalmesh", "meshpy",
                      "scikit-image", "matplotlib", "h5py"],
    description="Various useful tools for meshing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=about["__license__"],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
