import glob
import logging
import os.path
import platform
import sys

from setuptools import Extension, setup

NAME = 'implicit'
VERSION = '0.2.6'
SRC_ROOT = 'implicit'

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

is_dev = 'dev' in VERSION
if is_dev and not use_cython:
    raise RuntimeError('Cython required to build dev version of %s.' % NAME)

use_openmp = True


def define_extensions(use_cython=False):
    if sys.platform.startswith("win"):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ['/O2', '/openmp']
        link_args = []
    else:
        compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']
        link_args = []
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

    src_ext = '.pyx' if use_cython else '.cpp'
    modules = [Extension("implicit." + name,
                         [os.path.join("implicit", name + src_ext)],
                         language='c++',
                         extra_compile_args=compile_args, extra_link_args=link_args)
               for name in ['_als', '_nearest_neighbours']]

    if use_cython:
        return cythonize(modules)
    else:
        return modules


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def set_gcc():
    """
    Try to find and use GCC on OSX for OpenMP support.
    """
    # For macports and homebrew
    patterns = ['/opt/local/bin/g++-mp-[0-9].[0-9]',
                '/opt/local/bin/g++-mp-[0-9]',
                '/usr/local/bin/g++-[0-9].[0-9]',
                '/usr/local/bin/g++-[0-9]']

    if 'darwin' in platform.platform().lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()

        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


set_gcc()

setup(
    name=NAME,
    version=VERSION,
    description='Collaborative Filtering for Implicit Datasets',
    url='http://github.com/benfred/implicit/',
    author='Ben Frederickson',
    author_email='ben@benfrederickson.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'],

    keywords='Matrix Factorization, Implicit Alternating Least Squares, '
             'Collaborative Filtering, Recommender Systems',

    packages=[SRC_ROOT],
    install_requires=['numpy', 'scipy>=0.16'],
    setup_requires=["Cython >= 0.19"],
    ext_modules=define_extensions(use_cython),
    test_suite="tests",
)
