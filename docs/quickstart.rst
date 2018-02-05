.. Implicit documentation master file, created by
   sphinx-quickstart on Mon Jul 10 17:23:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Implicit
====================================

Fast Python Collaborative Filtering for Implicit Datasets

This project provides fast Python implementations of several different popular recommendation algorithms for
implicit feedback datasets:

 * Alternating Least Squares as described in the papers `Collaborative Filtering for Implicit Feedback Datasets <http://yifanhu.net/PUB/cf.pdf>`_ and in `Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering <https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf>`_.
 * `Bayesian Personalized Ranking <https://arxiv.org/pdf/1205.2618.pdf>`_
 * Item-Item Nearest Neighbour models, using Cosine, TFIDF or BM25 as a distance metric

All models have multi-threaded training routines, using Cython and OpenMP to fit the models in
parallel among all available CPU cores.  In addition, the ALS and BPR models both have custom CUDA
kernels - enabling fitting on compatible GPU's. This library also supports using approximate nearest neighbours libraries such as `Annoy <https://github.com/spotify/annoy>`_, `NMSLIB <https://github.com/searchivarius/nmslib>`_
and `Faiss <https://github.com/facebookresearch/faiss>`_ for `speeding up making recommendations <http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/>`_.


Installation
------------

To install:

``pip install implicit``

Basic Usage
-----------

.. code-block:: python

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


Articles about Implicit
-----------------------


These blog posts describe the algorithms that power this library:

 * `Finding Similar Music with Matrix Factorization <http://www.benfrederickson.com/matrix-factorization>`_
 * `Faster Implicit Matrix Factorization <http://www.benfrederickson.com/fast-implicit-matrix-factorization>`_
 * `Implicit Matrix Factorization on the GPU <http://www.benfrederickson.com/implicit-matrix-factorization-on-the-gpu/>`_
 * `Approximate Nearest Neighbours for Recommender Systems <http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/>`_
 * `Distance Metrics for Fun and Profit <http://www.benfrederickson.com/distance-metrics/>`_

There are also several other blog posts about using Implicit to build recommendation systems:

 * `Recommending GitHub Repositories with Google BigQuery and the implicit library <https://medium.com/@jbochi/recommending-github-repositories-with-google-bigquery-and-the-implicit-library-e6cce666c77>`_
 * `Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models <http://blog.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/>`_
 * `A Gentle Introduction to Recommender Systems with Implicit Feedback <https://jessesw.com/Rec-System/>`_

Requirements
------------

This library requires SciPy version 0.16 or later. Running on OSX requires an OpenMP compiler,
which can be installed with homebrew: ``brew install gcc``.

