import os
from os import path
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pt-snnl",
    version="1.0.0",
    packages=["snnl", "snnl.utils"],
    url="https://github.com/AFAgarap/pt-snnl",
    license="AGPL-3.0 License",
    author="Abien Fred Agarap",
    author_email="abienfred.agarap@gmail.com",
    description="PyTorch package for soft nearest neighbor loss function.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.19.4",
        "torchvision==0.8.2",
        "torch==1.7.1",
        "pt_datasets==0.9.9",
    ],
)
