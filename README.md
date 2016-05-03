Implicit [![Build Status](https://travis-ci.org/benfred/implicit.svg?branch=master)](https://travis-ci.org/benfred/implicit)
=======

Fast Python Collaborative Filtering for Implicit Datasets.

This project provides a fast Python implementation of the algorithm described in the paper [Collaborative Filtering for Implicit Feedback Datasets](
http://yifanhu.net/PUB/cf.pdf).


To install:

```
pip install implicit
```

Basic usage:

```python
import implicit
user_factors, item_factors = implicit.alternating_least_squares(data, factors=50)
```

The examples folder has a program showing how to use this to [compute similar artists on the
last.fm dataset](https://github.com/benfred/implicit/blob/master/examples/lastfm.py).

#### Requirements

This library requires SciPy version 0.16 or later. Running on OSX requires an OpenMP compiler,
which can be installed with homebrew: ```brew install gcc```. 

#### Why Use This?

This library came about because I was looking for an efficient Python
implementation of this algorithm for a [blog
post on matrix factorization](http://www.benfrederickson.com/matrix-factorization/). The other python
packages were too slow, and integrating with a different language or framework was too cumbersome.

The core of this package is written in Cython, leveraging OpenMP to
parallelize computation. Linear Algebra is done using the BLAS and LAPACK
libraries distributed with SciPy. This leads to extremely fast matrix factorization.

On a simple [benchmark](https://github.com/benfred/implicit/blob/master/examples/benchmark.py), this
library is about 1.8 times faster than the multithreaded C++ implementation provided by Quora's
[QMF Library](https://github.com/quora/qmf) and at least 60,000 times faster than [implicit-mf](https://github.com/MrChrisJohnson/implicit-mf).

This library has been tested with Python 2.7 and 3.5. Running 'tox' will
run unittests on both versions, and verify that all python files pass flake8.

#### Optimal Configuration

I'd recommend configure SciPy to use Intel's MKL matrix libraries. One easy way of doing this is by installing the Anaconda Python distribution.

For systems using OpenBLAS, I highly recommend setting 'export OPENBLAS_NUM_THREADS=1'. This disables its internal multithreading ability, which leads to
substantial speedups for this package.

Released under the MIT License
