#include "mag2_call.cuh"
#include <cuda_runtime.h>

__global__ void compute_mag2_kernel(const float2* input, float* output, const int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride)
        output[i] = input[i].x * input[i].x + input[i].y * input[i].y; 
}

void compute_mag2(float2* host_input, float* host_output, int N)
{
    float2* device_input;
    float* device_output;
    size_t device_input_size_bytes = N * sizeof(float2);
    size_t device_output_size_bytes = N * sizeof(float);
    cudaErrorCheck(cudaMalloc((void**)&device_input, device_input_size_bytes), "device_input malloc");
    cudaErrorCheck(cudaMalloc((void**)&device_output, device_output_size_bytes), "device_output malloc");
    cudaErrorCheck(cudaMemcpy(device_input, host_input, device_input_size_bytes, cudaMemcpyHostToDevice), "input host->device copy");
    
    int THREADS_PER_BLOCK = 256;
    int gridSize = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_mag2_kernel<<<gridSize, THREADS_PER_BLOCK>>>(device_input, device_output, N);
    cudaErrorCheck(cudaGetLastError(), "Lauch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Sync");

    cudaErrorCheck(cudaMemcpy(host_output, device_output, device_output_size_bytes, cudaMemcpyDeviceToHost), "output device->host copy");

    cudaErrorCheck(cudaFree(device_input), "device_input free");
    cudaErrorCheck(cudaFree(device_output), "device_output free");
}
