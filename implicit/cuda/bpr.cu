#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>


#include "implicit/cuda/als.h"
#include "implicit/cuda/utils.cuh"

namespace implicit {
__global__ void bpr_update_kernel(int samples, unsigned int * random_likes, unsigned int * random_dislikes,
                                  int * itemids, int * userids,
                                  int item_count, int user_count, int factors,
                                  float * X, float * Y,
                                  float learning_rate, float reg,
                                  unsigned int * stats) {
    extern __shared__ float shared_memory[];
    float * temp = &shared_memory[0];

    int correct = 0;

    for (int i = blockIdx.x; i < samples; i += gridDim.x) {
        int liked_index = random_likes[i] % samples,
            disliked_index = random_dislikes[i] % samples;

        float * user = &X[userids[liked_index] * factors],
              * liked = &Y[itemids[liked_index] * factors],
              * disliked = &Y[itemids[disliked_index] * factors];

        float user_val = user[threadIdx.x],
              liked_val = liked[threadIdx.x],
              disliked_val = disliked[threadIdx.x];

        temp[threadIdx.x] = liked_val - disliked_val;
        float score = dot(user, temp);
        float z = 1.0 / (1.0 + exp(score));

        if (z < .5) correct++;

        liked[threadIdx.x]    += learning_rate * ( z * user_val - reg * liked_val);
        disliked[threadIdx.x] += learning_rate * (-z * user_val - reg * disliked_val);
        // We're storing the item bias in the last column of the matrix - with the user = 1
        // in that column. Don't update the user value in that case
        if (threadIdx.x < factors ){
            user[threadIdx.x] += learning_rate * ( z * (liked_val - disliked_val) - reg * user_val);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(stats, correct);
    }
}

#define CHECK_CURAND(code) { checkCurand((code), __FILE__, __LINE__); }
inline void checkCurand(curandStatus_t code, const char *file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        std::stringstream err;
        err << "CURAND error: " << code << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}

int bpr_update(const CudaCOOMatrix & Ciu,
               CudaDenseMatrix * X,
               CudaDenseMatrix * Y,
               float learning_rate, float reg, long seed) {
    int item_count = Y->rows, user_count = X->rows, factors = X->cols;
    if (X->cols != Y->cols) throw std::invalid_argument("X and Y should have the same number of columns");
    if (Ciu.cols != X->rows) throw std::invalid_argument("Dimensionality mismatch between Ciu and X");
    if (Ciu.rows != Y->rows) throw std::invalid_argument("Dimensionality mismatch between Ciu and Y");

    // allocate some memory
    unsigned int * stats;
    CHECK_CUDA(cudaMalloc(&stats, sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(stats, 0, sizeof(unsigned int)));

    // initialize memory for randomly picked positive/negative items
    unsigned int * random_likes, * random_dislikes;
    CHECK_CUDA(cudaMalloc(&random_likes, Ciu.nonzeros * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&random_dislikes, Ciu.nonzeros * sizeof(unsigned int)));

    // Create a seeded RNG
    curandGenerator_t rng;
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));

    // Randomly pick values
    CHECK_CURAND(curandGenerate(rng, random_likes, Ciu.nonzeros));
    CHECK_CURAND(curandGenerate(rng, random_dislikes, Ciu.nonzeros));

    // TODO: multi-gpu support
    int devId;
    CHECK_CUDA(cudaGetDevice(&devId));

    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                      cudaDevAttrMultiProcessorCount,
                                      devId));

    int block_count = 128 * multiprocessor_count;
    int thread_count = factors;
    int shared_memory_size = sizeof(float) * (factors);

    bpr_update_kernel<<<block_count, thread_count, shared_memory_size>>>(
        Ciu.nonzeros, random_likes, random_dislikes,
        Ciu.row, Ciu.col, item_count, user_count, factors,
        X->data, Y->data, learning_rate, reg, stats);

    CHECK_CUDA(cudaDeviceSynchronize());

    // we're returning the number of correctly ranked items, get that value from the device
    unsigned int correct = 0;
    CHECK_CUDA(cudaMemcpy(&correct, stats, sizeof(correct), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(random_likes));
    CHECK_CUDA(cudaFree(random_dislikes));
    CHECK_CUDA(cudaFree(stats));
    return correct;
}
}
