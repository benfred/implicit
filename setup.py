import glob
import os.path
import platform
import sys

from setuptools import Command, Extension, setup


def define_extensions(cythonize=False):
    compile_args = ['-fopenmp', '-ffast-math']

    if 'anaconda' not in sys.version.lower():
        compile_args.append('-march=native')

    if cythonize:
        implicit_cython = "implicit/_implicit.pyx"
    else:
        implicit_cython = "implicit/_implicit.c"

    return [Extension("implicit._implicit", [implicit_cython],
                      extra_link_args=["-fopenmp"],
                      extra_compile_args=compile_args)]


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def set_gcc():
    """
    Try to find and use GCC on OSX for OpenMP support.
    """
    # For macports and homebrew
    patterns = ['/opt/local/bin/gcc-mp-[0-9].[0-9]',
                '/opt/local/bin/gcc-mp-[0-9]',
                '/usr/local/bin/gcc-[0-9].[0-9]',
                '/usr/local/bin/gcc-[0-9]']

    if 'darwin' in platform.platform().lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()

        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ["CC"] = gcc

        else:
            raise Exception('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


set_gcc()


class Cythonize(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from Cython.Build import cythonize
        cythonize(define_extensions(cythonize=True))


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='implicit',
    version="0.1.1",
    description='Collaborative Filtering for Implicit Datasets',
    long_description=long_description,
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
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],

    keywords='Matrix Factorization, Implicit Alternating Least Squares, '
             'Collaborative Filtering, Recommender Systems',

    packages=['implicit'],
    install_requires=['numpy', 'scipy>=0.16'],
    cmdclass={'cythonize': Cythonize},
    setup_requires=["Cython >= 0.19"],
    ext_modules=define_extensions(),
    test_suite="tests",
)
