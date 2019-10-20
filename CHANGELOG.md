## v0.3.9

* Add ability to pickle nearest neighbours recommenders (#191)[https://github.com/benfred/implicit/issues/191]
* add NDCG method to evaluation (#212)[https://github.com/benfred/implicit/pull/212]
* Add a 'recommend_all' method for matrix factorization models (#179[https://github.com/benfred/implicit/pull/179]

## v0.3.8

* Ensure progress bar hits 100% during xval
* Fix bm25recommender missing default parameter on fit

## v0.3.7

* Fix GPU faiss model with > 1024 results (#149)[https://github.com/benfred/implicit/issues/149]
* Add a reddit votes dataseet
* Add similar users calculation in MF modeles (#139)[https://github.com/benfred/implicit/pull/139]
* Add an option to whether to include previously liked items or not (#131)[https://github.com/benfred/implicit/issues/131]
* Add option for negative preferences to ALS modele (#119)[https://github.com/benfred/implicit/issues/119)
* Add filtering negative feedback in test set (#124)[https://github.com/benfred/implicit/issues/124)

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
