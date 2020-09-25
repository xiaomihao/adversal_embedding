import os

import setuptools
from setuptools import setup

install_requires = [
    "numpy",
    "keras",
    "tensorflow==1.15.0",
]


setup(
    # _PKG_NAME will be used in Makefile for dev release
    name=os.getenv("_PKG_NAME", "adversal_embedding"),
    version="0.0.1",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/xiaomihao/adversal_embedding",
    license="Apache 2.0",
    author="Xiao Mi",
    author_email="1922188869@qq.com",
    description="adversal_embedding",
    install_requires=install_requires,
)
