import io
import os.path

from setuptools import find_packages
from skbuild import setup


def read(file_name):
    """Read a text file and return the content as a string."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with io.open(file_path, encoding="utf-8") as f:
        return f.read()


def exclude_non_implicit_cmake_files(cmake_manifest):
    # we seem to be picking up a bunch of unrelated files from thrust/spdlog/rmm
    # filter the cmake manifest down to things from this package only
    return [f for f in cmake_manifest if "implicit" in f]


setup(
    name="implicit",
    version="0.7.2",
    description="Collaborative Filtering for Implicit Feedback Datasets",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="http://github.com/benfred/implicit/",
    author="Ben Frederickson",
    author_email="ben@benfrederickson.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=(
        "Matrix Factorization, Implicit Alternating Least Squares, "
        "Collaborative Filtering, Recommender Systems"
    ),
    packages=find_packages(),
    install_requires=["numpy>=1.17.0", "scipy>=0.16", "tqdm>=4.27", "threadpoolctl"],
    cmake_process_manifest_hook=exclude_non_implicit_cmake_files,
)
