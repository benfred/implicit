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
