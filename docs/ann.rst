Approximate Alternating Least Squares
=====================================

This library supports using a couple of different approximate nearest neighbours libraries
to speed up the recommend and similar_items methods of the AlternatingLeastSquares model.

The potential speedup of using these methods can be quite significant, at the risk of
potentially missing relevant results:

.. image:: recommendperf.png

See `this post comparing the different ANN libraries 
<http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/>`_ for
more details.

NMSLibAlternatingLeastSquares
-----------------------------
.. autoclass:: implicit.approximate_als.NMSLibAlternatingLeastSquares
   :members:

AnnoyAlternatingLeastSquares
----------------------------
.. autoclass:: implicit.approximate_als.AnnoyAlternatingLeastSquares
   :members:

FaissAlternatingLeastSquares
-----------------------------
.. autoclass:: implicit.approximate_als.FaissAlternatingLeastSquares
   :members:
