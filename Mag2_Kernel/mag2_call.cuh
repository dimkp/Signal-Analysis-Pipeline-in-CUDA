#include <stdio.h>
#include <cuda_runtime.h>

static void cudaErrorCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s -> %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

void compute_mag2(float2* h_in, float* h_out, int N);