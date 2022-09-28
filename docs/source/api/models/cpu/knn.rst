Nearest Neighbour Models
========================

Implicit contains several item-item nearest neighbour models. See
`this post <https://www.benfrederickson.com/distance-metrics/>`_ for more information.


CosineRecommender
-----------------

.. autoclass:: implicit.nearest_neighbours.CosineRecommender
   :members:
   :show-inheritance:
   :exclude-members: fit,recommend,similar_items,similar_users

TFIDFRecommender
-------------------

.. autoclass:: implicit.nearest_neighbours.TFIDFRecommender
   :members:
   :show-inheritance:
   :exclude-members: fit,recommend,similar_items,similar_users

BM25Recommender
---------------

.. autoclass:: implicit.nearest_neighbours.BM25Recommender
   :members:
   :show-inheritance:
   :exclude-members: fit,recommend,similar_items,similar_users


ItemItemRecommender
-------------------

.. autoclass:: implicit.nearest_neighbours.ItemItemRecommender
   :members:
   :show-inheritance:
   :exclude-members: fit,recommend,similar_items,similar_users
