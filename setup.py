import os
import shutil
from setuptools import setup
from os import path
import numpy

# clean previous build
for root, dirs, files in os.walk("./QuantifAI/", topdown=False):
    for name in dirs:
        if name == "build":
            shutil.rmtree(name)

this_directory = path.abspath(path.dirname(__file__))


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


long_description = read_file(".pip_readme.rst")
required = read_requirements("requirements/requirements-core.txt")

include_dirs = [
    numpy.get_include(),
]

extra_link_args = []

setup(
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    name="quantifai",
    version="1.0.0",
    prefix=".",
    url="https://github.com/tobias-liaudat/QuantifAI",
    author="Tobias Liaudat et al.",
    author_email="tobiasliaudat@gmail.com",
    license="MIT license",
    install_requires=required,
    description="Scalable Bayesian uncertainty quantification with data-driven priors for radio interferometric imaging",
    long_description_content_type="text/x-rst",
    long_description=long_description,
    packages=["quantifai"],
)
