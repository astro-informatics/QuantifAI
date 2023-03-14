import os
import shutil
from setuptools import setup

import numpy

# clean previous build
for root, dirs, files in os.walk("./large_scale_UQ/", topdown=False):
    for name in dirs:
        if (name == "build"):
            shutil.rmtree(name)

from os import path
this_directory = path.abspath(path.dirname(__file__))

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

def read_file(file):
   with open(file) as f:
        return f.read()

# long_description = read_file(".pip_readme.rst")
required = read_requirements("requirements/requirements-core.txt")

include_dirs = [numpy.get_include(),]

extra_link_args=[]

setup(
    classifiers=['Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Operating System :: OS Independent',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research'
                 ],
    name = "large_scale_UQ",
    version = "0.0.1",
    prefix='.',
    url='https://github.com/tobias-liaudat/large-scale-UQ',
    author='Tobias Liaudat et al.',
    author_email='tobiasliaudat@gmail.com',
    license='GNU General Public License v3 (GPLv3)',
    install_requires=required,
    description='Large scale convex uncertainty quantification',
    # long_description_content_type = "text/x-rst",
    # long_description=long_description,
    packages=['large_scale_UQ']
)
