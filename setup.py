import io
import os.path

from setuptools import find_packages
from skbuild import setup

try:
    import numpy.distutils  # noqa
except ImportError:
    pass


def read(file_name):
    """Read a text file and return the content as a string."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with io.open(file_path, encoding="utf-8") as f:
        return f.read()


setup(
    name="implicit",
    version="0.4.8",
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
    install_requires=["numpy", "scipy>=0.16", "tqdm>=4.27"],
)
