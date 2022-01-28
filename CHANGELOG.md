## v0.5.0

### Breaking API Changes

The API for implicit has substantially changed in v0.5.0 - and any code written for the previous
API will need to be rewritten:

* Change model.fit to take a user_items sparse matrix [#484](https://github.com/benfred/implicit/pull/484)
* Return numpy arrays from recommend methods [#482](https://github.com/benfred/implicit/pull/482)
* Don't require empty rows in user_items and item_users parameters [#526](https://github.com/benfred/implicit/pull/526)
* Unify API for rank_items/recommend/recommend_all [#489](https://github.com/benfred/implicit/issues/489)

### Performance Improvements

* Speedup evaluation by using batch recommend functions [#520](https://github.com/benfred/implicit/pull/520)
* Use FAISS for GPU inference [#506](https://github.com/benfred/implicit/pull/506)
* Multithreaded speedups for CPU models [#517](https://github.com/benfred/implicit/pull/517)
* Use thrust::binary_search to verify negative samples on GPU [#524](https://github.com/benfred/implicit/pull/524)
* Release GIL on GPU calls [#528](https://github.com/benfred/implicit/pull/528)

### New Features

* Add incremental retraining support for ALS models [#527](https://github.com/benfred/implicit/pull/527)
* Add filtering for similar_items and similar_users [#488](https://github.com/benfred/implicit/issues/488)
* Add support for recalculate_users/items on the GPU [#515](https://github.com/benfred/implicit/pull/515)
* Add methods for converting MF models to/from gpu [#521](https://github.com/benfred/implicit/pull/521)
* Add a tutorial notebook for the lastfm example [#529](https://github.com/benfred/implicit/pull/529)
* Approximate nearest neighbour for BPR/LMF and GPU models [#487](https://github.com/benfred/implicit/issues/487)
* Dynamically detect CUDA availability [#174](https://github.com/benfred/implicit/issues/174)

## v0.4.5

* GPU Inference [#406](https://github.com/benfred/implicit/pull/406)
* Fix ALS model for case of > 2^31 interactions [#400](https://github.com/benfred/implicit/pull/400)
* Fix GPU dot product when the # of factors wasn't warp aligned [#427](https://github.com/benfred/implicit/pull/427)
* Use gpu registers for dot product [#448](https://github.com/benfred/implicit/pull/448)
* Fix random output with LMF/BPR models [#408](https://github.com/benfred/implicit/pull/408)
* Add seed for test-train split [#411](https://github.com/benfred/implicit/pull/411)

## v0.4.4

* Adds Precompiled CUDA packages on conda-forge
* Drops support for CUDA 8

## v0.4.3

* Implement filter_already_liked_items option [#328](https://github.com/benfred/implicit/pull/328)
* Fix bug in ALS explain when user_items contains negative confidence values [#313](https://github.com/benfred/implicit/pull/313)
* Improve numerical stability of LMF [#383](https://github.com/benfred/implicit/pull/383)
* Add error check after training for NaN factors [#381](https://github.com/benfred/implicit/pull/381)
* Support building with Cuda 11

## v0.4.0

* Add logistic matrix factorization algorithm [#231](https://github.com/benfred/implicit/pull/231)
* Use tqdm for progress bars [#240](https://github.com/benfred/implicit/pull/240)
* Add AUC@K [#275](https://github.com/benfred/implicit/pull/275)

## v0.3.9

* Add ability to pickle nearest neighbours recommenders [#191](https://github.com/benfred/implicit/issues/191)
* add NDCG method to evaluation [#212](https://github.com/benfred/implicit/pull/212)
* Add a 'recommend_all' method for matrix factorization models [#179](https://github.com/benfred/implicit/pull/179)

## v0.3.8

* Ensure progress bar hits 100% during xval
* Fix bm25recommender missing default parameter on fit

## v0.3.7

* Fix GPU faiss model with > 1024 results [#149](https://github.com/benfred/implicit/issues/149)
* Add a reddit votes dataseet
* Add similar users calculation in MF modeles [#139](https://github.com/benfred/implicit/pull/139)
* Add an option to whether to include previously liked items or not [#131](https://github.com/benfred/implicit/issues/13)
* Add option for negative preferences to ALS modele [#119](https://github.com/benfred/implicit/issues/119)
* Add filtering negative feedback in test set [#124](https://github.com/benfred/implicit/issues/124)

## v0.3.6

* Adds evaluation functionality with functions for computing P@k and MAP@K and generating a train/test split
* BPR model now verifies negative samples haven’t been actually liked now, leading to more accurate recommendations
* Faster KNN recommendations (up to 10x faster recommend calls)
* Various fixes for models when fitting on the GPU
* Fix CUDA install on Windows
* Display progress bars when fitting models using tqdm
* More datasets: added million song dataset, sketchfab, movielens 100k, 1m and 10m

## v0.3.5

* Use HDF5 files for distributing datasets
* Add rank_items method to recommender

## v0.3.3

* Fix issue with last user having no ratings in BPR model

## v0.3.2

* Support more than 2^31 training examples in ALS and BPR models
* Allow 64 bit factors for BPR

## v0.3.0
* Add a Bayesian Personalized Ranking model, with an option for fitting on the GPU

## v0.2.7
* Add Support for ANN libraries likes Faiss, NMSLIB and Annoy for making recommendations
