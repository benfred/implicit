import glob
import io
import logging
import os.path
import platform
import sys

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

from cuda_setup import CUDA, build_ext

NAME = "implicit"
VERSION = "0.4.4"


use_openmp = True


def define_extensions():
    if sys.platform.startswith("win"):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ["/O2", "/openmp"]
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = "/usr/local/opt/gcc/lib/gcc/" + gcc[-1] + "/"
            link_args = ["-Wl,-rpath," + rpath]
        else:
            link_args = []

        compile_args = ["-Wno-unused-function", "-Wno-maybe-uninitialized", "-O3", "-ffast-math"]
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

        compile_args.append("-std=c++11")
        link_args.append("-std=c++11")

    # we need numpy to build so we can include the arrayobject.h in the .cpp builds
    # try:
    #     import numpy as np
    # except ImportError:
    #     raise ValueError("numpy is required to build from source")

    src_ext = ".pyx"
    modules = [
        Extension(
            "implicit." + name,
            [os.path.join("implicit", name + src_ext)],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
        for name in ["_nearest_neighbours", "lmf", "evaluation"]
    ]
    modules.extend(
        [
            Extension(
                "implicit.cpu." + name,
                [os.path.join("implicit", "cpu", name + src_ext)],
                language="c++",
                extra_compile_args=compile_args,
                extra_link_args=link_args,
            )
            for name in ["_als", "bpr"]
        ]
    )
    modules.append(
        Extension(
            "implicit." + "recommender_base",
            [
                os.path.join("implicit", "recommender_base" + src_ext),
                os.path.join("implicit", "topnc.cpp"),
            ],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    )

    if CUDA:
        conda_prefix = os.getenv("CONDA_PREFIX")
        include_dirs = [CUDA["include"], "."]
        library_dirs = [CUDA["lib64"]]
        if conda_prefix:
            include_dirs.append(os.path.join(conda_prefix, "include"))
            library_dirs.append(os.path.join(conda_prefix, "lib"))

        modules.append(
            Extension(
                "implicit.gpu._cuda",
                [
                    os.path.join("implicit", "gpu", "_cuda" + src_ext),
                    os.path.join("implicit", "gpu", "als.cu"),
                    os.path.join("implicit", "gpu", "bpr.cu"),
                    os.path.join("implicit", "gpu", "matrix.cu"),
                    os.path.join("implicit", "gpu", "device_buffer.cu"),
                    os.path.join("implicit", "gpu", "random.cu"),
                    os.path.join("implicit", "gpu", "knn.cu"),
                ],
                language="c++",
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                library_dirs=library_dirs,
                libraries=["cudart", "cublas", "curand"],
                include_dirs=include_dirs,
            )
        )
    else:
        print("Failed to find CUDA toolkit. Building without GPU acceleration.")

    return cythonize(modules)


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python


def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9]*.[0-9]*",
        "/opt/local/bin/g++-mp-[0-9]*",
        "/usr/local/bin/g++-[0-9]*.[0-9]*",
        "/usr/local/bin/g++-[0-9]*",
    ]
    if platform.system() == "Darwin":
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew
    if platform.system() == "Darwin":
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning(
                "No GCC available. Install gcc from Homebrew " "using brew install gcc."
            )


set_gcc()


def read(file_name):
    """Read a text file and return the content as a string."""
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with io.open(file_path, encoding="utf-8") as f:
        return f.read()


setup(
    name=NAME,
    version=VERSION,
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
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="Matrix Factorization, Implicit Alternating Least Squares, "
    "Collaborative Filtering, Recommender Systems",
    packages=find_packages(),
    install_requires=["numpy", "scipy>=0.16", "tqdm>=4.27"],
    setup_requires=["Cython>=0.24"],
    ext_modules=define_extensions(),
    cmdclass={"build_ext": build_ext},
    test_suite="tests",
)
