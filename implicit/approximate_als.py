""" Models that use various Approximate Nearest Neighbours libraries in order to quickly
generate recommendations and lists of similar items.

See http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/
"""
import implicit.gpu


def NMSLibAlternatingLeastSquares(
    *args,
    approximate_similar_items=True,
    approximate_recommend=True,
    method="hnsw",
    index_params=None,
    query_params=None,
    use_gpu=implicit.gpu.HAS_CUDA,
    **kwargs
):
    # delay importing here in case nmslib isn't installed
    from implicit.ann.nmslib import NMSLibModel

    # note that we're using the factory function here to instantiate a CPU/GPU model as appropriate
    als_model = implicit.als.AlternatingLeastSquares(*args, use_gpu=use_gpu, **kwargs)
    return NMSLibModel(
        als_model,
        approximate_similar_items=approximate_similar_items,
        approximate_recommend=approximate_recommend,
        method=method,
        index_params=index_params,
        query_params=query_params,
    )


def AnnoyAlternatingLeastSquares(
    *args,
    approximate_similar_items=True,
    approximate_recommend=True,
    n_trees=50,
    search_k=-1,
    use_gpu=implicit.gpu.HAS_CUDA,
    **kwargs
):
    als_model = implicit.als.AlternatingLeastSquares(*args, use_gpu=use_gpu, **kwargs)
    from implicit.ann.annoy import AnnoyModel

    return AnnoyModel(
        als_model,
        approximate_similar_items=approximate_similar_items,
        approximate_recommend=approximate_recommend,
        n_trees=n_trees,
        search_k=search_k,
    )


def FaissAlternatingLeastSquares(
    *args,
    approximate_similar_items=True,
    approximate_recommend=True,
    nlist=400,
    nprobe=20,
    use_gpu=implicit.gpu.HAS_CUDA,
    **kwargs
):
    # note that we're using the factory function here to instantiate a CPU/GPU model as appropriate
    als_model = implicit.als.AlternatingLeastSquares(*args, use_gpu=use_gpu, **kwargs)

    from implicit.ann.faiss import FaissModel

    return FaissModel(
        als_model,
        approximate_similar_items=approximate_similar_items,
        approximate_recommend=approximate_recommend,
        nlist=nlist,
        nprobe=nprobe,
        use_gpu=use_gpu,
    )
