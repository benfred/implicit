Implicit
=======

[![Build Status](https://travis-ci.org/benfred/implicit.svg?branch=master)](https://travis-ci.org/benfred/implicit)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/9kfbvx5i6dc48yr0?svg=true)](https://ci.appveyor.com/project/benfred/implicit)

Fast Python Collaborative Filtering for Implicit Datasets.

This project provides fast Python implementations of the algorithms described in the paper [Collaborative Filtering for Implicit Feedback Datasets](
http://yifanhu.net/PUB/cf.pdf) and in [Applications of the Conjugate Gradient Method for Implicit
Feedback Collaborative
Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf).


To install:

```
pip install implicit
```

Basic usage:

```python
import implicit

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(item_user_data)

# recommend items for a user
user_items = item_user_data.T.tocsr()
recommendations = model.recommend(userid, user_items)

# find related items
related = model.similar_items(itemid)
```

The examples folder has a program showing how to use this to [compute similar artists on the
last.fm dataset](https://github.com/benfred/implicit/blob/master/examples/lastfm.py).

For more information see the [documentation](http://implicit.readthedocs.io/).

#### Articles about Implicit

Several posts have been written talking about using Implicit to build recommendation systems:

 * [Recommending GitHub Repositories with Google BigQuery and the implicit library](https://medium.com/@jbochi/recommending-github-repositories-with-google-bigquery-and-the-implicit-library-e6cce666c77)
 * [Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models](http://blog.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/)
 * [A Gentle Introduction to Recommender Systems with Implicit Feedback](https://jessesw.com/Rec-System/).

There are also a couple posts talking about the algorithms that power this library:

 * [Faster Implicit Matrix Factorization](http://www.benfrederickson.com/fast-implicit-matrix-factorization)
 * [Finding Similar Music with Matrix Factorization](http://www.benfrederickson.com/matrix-factorization)
 * [Distance Metrics for Fun and Profit](http://www.benfrederickson.com/distance-metrics/)

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
[QMF Library](https://github.com/quora/qmf) and at least 60,000 times faster than
[implicit-mf](https://github.com/MrChrisJohnson/implicit-mf).

A [follow up post](http://www.benfrederickson.com/fast-implicit-matrix-factorization/) describes
further performance improvements based on the Conjugate Gradient method - that further boosts performance
by 3x to over 19x depending on the number of factors used.

This library has been tested with Python 2.7 and 3.5. Running 'tox' will
run unittests on both versions, and verify that all python files pass flake8.

#### Optimal Configuration

I'd recommend configure SciPy to use Intel's MKL matrix libraries. One easy way of doing this is by installing the Anaconda Python distribution.

For systems using OpenBLAS, I highly recommend setting 'export OPENBLAS_NUM_THREADS=1'. This disables its internal multithreading ability, which leads to
substantial speedups for this package.

Released under the MIT License
