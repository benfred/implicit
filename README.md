implicit [![Build Status](https://travis-ci.org/benfred/implicit.svg?branch=master)](https://travis-ci.org/benfred/implicit)
=======

Fast Python Collaborative Filtering for Implicit Datasets.

This project provides a fast Python implementation of the algorithm decribed in the paper [Collaborative Filtering for Implicit Feedback Datasets](
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

#### Requirements

This library requires SciPy version 0.16 or later.

#### Why Use This?

This library came about because I was looking for an efficient  Python
implementation of this algorithm for a blog post I am writing.

The other [pure python implementation](https://github.com/MrChrisJohnson/implicit-mf) was much too slow on the dataset I'm interested in: this package finishes factorizing the last.fm dataset in about 10 minutes (50 factors, 15 iterations, 2015
macbook pro) where I estimate that the implicit-mf package would take 250 days
or so to do the same computation.

The core of this package is written in Cython, leveraging OpenMP to
parallelize computation. Linear Algebra is done using the BLAS and LAPACK
libraries distributed with SciPy. There also exists a pure python
implementation as a reference.

This library has been tested with Python 2.7 and 3.5. Running 'tox' will
run unittests on both versions, and verify that all python files pass flake8.

#### TODO

This is still a work in progress. Things immediately on the horizon:

- Example application
- Sphinx autodoc
- Test on linux, verify openmp support actually works
- Benchmark

Released under the MIT License
