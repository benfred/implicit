import glob
import logging
import os.path
import platform
import sys

from setuptools import Extension, setup, find_packages

from cuda_setup import CUDA, build_ext


NAME = 'implicit'
VERSION = "0.3.7"

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
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = '/usr/local/opt/gcc/lib/gcc/' + gcc[-1] + '/'
            link_args = ['-Wl,-rpath,' + rpath]
        else:
            link_args = []

        compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']
        if use_openmp:
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")

        compile_args.append("-std=c++11")
        link_args.append("-std=c++11")

    src_ext = '.pyx' if use_cython else '.cpp'
    modules = [Extension("implicit." + name,
                         [os.path.join("implicit", name + src_ext)],
                         language='c++',
                         extra_compile_args=compile_args, extra_link_args=link_args)
               for name in ['_als', '_nearest_neighbours', 'bpr', 'evaluation']]

    if CUDA:
        modules.append(Extension("implicit.cuda._cuda",
                                 [os.path.join("implicit", "cuda", "_cuda" + src_ext),
                                  os.path.join("implicit", "cuda", "als.cu"),
                                  os.path.join("implicit", "cuda", "bpr.cu"),
                                  os.path.join("implicit", "cuda", "matrix.cu")],
                                 language="c++",
                                 extra_compile_args=compile_args,
                                 extra_link_args=link_args,
                                 library_dirs=[CUDA['lib64']],
                                 libraries=['cudart', 'cublas', 'curand'],
                                 include_dirs=[CUDA['include'], '.']))
    else:
        print("Failed to find CUDA toolkit. Building without GPU acceleration.")

    if use_cython:
        return cythonize(modules)
    else:
        return modules


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
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
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew

    if 'darwin' in platform.platform().lower():
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


set_gcc()

try:
    # if we don't have pandoc installed, don't worry about it
    import pypandoc
    long_description = pypandoc.convert_file("README.md", "rst")
except ImportError:
    long_description = ''


setup(
    name=NAME,
    version=VERSION,
    description='Collaborative Filtering for Implicit Datasets',
    long_description=long_description,
    url='http://github.com/benfred/implicit/',
    author='Ben Frederickson',
    author_email='ben@benfrederickson.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
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

    packages=find_packages(),
    install_requires=['numpy', 'scipy>=0.16', 'tqdm'],
    setup_requires=["Cython>=0.24"],
    ext_modules=define_extensions(use_cython),
    cmdclass={'build_ext': build_ext},
    test_suite="tests",
)
