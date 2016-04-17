from setuptools import setup, Extension
from codecs import open
from os import path
import sys

from Cython.Distutils import build_ext
import numpy

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

compile_args = ['-O3', '-Wno-strict-prototypes', '-Wno-unused-function', '-Wno-unreachable-code']
link_args = []

if not sys.platform.startswith('darwin'):
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

extension = Extension(
    "implicit._implicit",
    ["implicit/_implicit.pyx"],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    include_dirs=[numpy.get_include()],
)

setup(
    name='implicit',
    version="0.1.0",
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
    setup_requires=["Cython >= 0.19"],
    ext_modules=[extension],
    test_suite="tests",
    cmdclass={'build_ext': build_ext},
)
