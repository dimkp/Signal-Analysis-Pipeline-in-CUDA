#include "radiometer.cuh"
#include <cuda_runtime.h>

__global__ void compute_mag2_kernel(const float2* __restrict__ input, float* __restrict__ output, const int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride)
        output[i] = input[i].x * input[i].x + input[i].y * input[i].y; 
}

__global__ void moving_average_kernel(const float* power, float* output, int N, int L)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float sum = 0.0f;
        int count = 0;

        for (int k = 0; k < L; k++)
        {
            int j = idx - k;
            if (j >= 0)
            {
                sum += power[j];
                count++;
            }
        }

        output[idx] = sum / count;
    }
}

void radiometer(float2* host_input, float* host_output, int N)
{
    float2* device_input;
    float* device_power;
    float* device_output;
    
    size_t device_input_size_bytes = N * sizeof(float2);
    size_t device_size_bytes = N * sizeof(float);
    
    cudaErrorCheck(cudaMalloc((void**)&device_input, device_input_size_bytes), "device_input malloc");
    cudaErrorCheck(cudaMalloc((void**)&device_power, device_size_bytes), "device_power malloc");
    cudaErrorCheck(cudaMalloc((void**)&device_output, device_size_bytes), "device_output malloc");

    cudaErrorCheck(cudaMemcpy(device_input, host_input, device_input_size_bytes, cudaMemcpyHostToDevice), "input host->device copy");
    
    // Kernel launch parameters
    int THREADS_PER_BLOCK = 256;
    int gridSize = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_mag2_kernel<<<gridSize, THREADS_PER_BLOCK>>>(device_input, device_power, N);
    cudaErrorCheck(cudaGetLastError(), "Lauch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Sync");




    
    // Final output of Averaged Squared Magnitude
    cudaErrorCheck(cudaMemcpy(host_output, device_output, device_size_bytes, cudaMemcpyDeviceToHost), "output device->host copy");

    // Memory cleanup
    cudaErrorCheck(cudaFree(device_input), "device_input free");
    cudaErrorCheck(cudaFree(device_power), "device_power free");
    cudaErrorCheck(cudaFree(device_output), "device_output free");
}
