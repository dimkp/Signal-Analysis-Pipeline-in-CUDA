#include "radiometer.cuh"
#include <cuda_runtime.h>

__global__ void compute_mag2_kernel(const float* input, float* output, const int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
        output[index] = input[index] * input[index]; 
}

void compute_mag2(float* host_input, float* host_output, int N)
{
    float* device_input, * device_output;
    size_t device_size_bytes = N * sizeof(float);
    cudaErrorCheck(cudaMalloc((void**)&device_input, device_size_bytes), "device_input malloc");
    cudaErrorCheck(cudaMalloc((void**)&device_output, device_size_bytes), "device_output malloc");
    cudaErrorCheck(cudaMemcpy(device_input, host_input, device_size_bytes, cudaMemcpyHostToDevice), "input host->device copy");
    
    int THREADS_PER_BLOCK = 256;
    int gridSize = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_mag2_kernel<<<gridSize, THREADS_PER_BLOCK>>>(device_input, device_output, N);
    cudaErrorCheck(cudaGetLastError(), "Lauch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Sync");

    cudaErrorCheck(cudaMemcpy(host_output, device_output, device_size_bytes, cudaMemcpyDeviceToHost), "output device->host copy");

    cudaErrorCheck(cudaFree(device_input), "device_input free");
    cudaErrorCheck(cudaFree(device_output), "device_output free");
}
