.. Implicit documentation master file, created by
   sphinx-quickstart on Mon Jul 10 17:23:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Implicit
====================================

Fast Python Collaborative Filtering for Implicit Datasets.

This project provides fast Python implementations of several different popular recommendation algorithms for
implicit feedback datasets:

 * Alternating Least Squares as described in the papers `Collaborative Filtering for Implicit Feedback Datasets <http://yifanhu.net/PUB/cf.pdf>`_ and in `Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering <https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf>`_.
 * `Bayesian Personalized Ranking <https://arxiv.org/pdf/1205.2618.pdf>`_
 * `Logistic Matrix Factorization <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>`_
 * Item-Item Nearest Neighbour models, using Cosine, TFIDF or BM25 as a distance metric

All models have multi-threaded training routines, using Cython and OpenMP to fit the models in
parallel among all available CPU cores.  In addition, the ALS and BPR models both have custom CUDA
kernels - enabling fitting on compatible GPU's. This library also supports using approximate nearest neighbours libraries such as `Annoy <https://github.com/spotify/annoy>`_, `NMSLIB <https://github.com/searchivarius/nmslib>`_
and `Faiss <https://github.com/facebookresearch/faiss>`_ for `speeding up making recommendations <http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Quickstart <quickstart>
    RecommenderBase <models>
    Alternating Least Squares <als>
    Bayesian Personalized Ranking <bpr>
    Logistic Matrix Factorization <lmf>
    Approximate Alternating Least Squares <ann>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
