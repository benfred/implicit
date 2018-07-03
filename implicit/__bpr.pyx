@cython.cdivision(True)
@cython.boundscheck(False)
def __bpr_update(RNGVector rng,
               integral[:] userids,
               integral[:] data, integral[:] itemids, integral[:] indptr,
               integral[:] pos_itemids, integral[:] pos_indptr,
               integral[:] neg_itemids, integral[:] neg_indptr,
               floating[:, :] X, floating[:, :] Y,
               float learning_rate, float reg, int num_threads,
               bool verify_neg):
