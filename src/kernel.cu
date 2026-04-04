#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <vector>
#include <ctime>
#include <fstream>
#include <filesystem>

#define PI 3.1415926535897932
#define THREADS_PER_BLOCK 256
#define MAX_FILTER_SIZE 32

static void cudaErrorCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s -> %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}


__global__ void FilteringKernel(const float* input, const float* filter, float* output, int N, int K)
{
    extern __shared__ float s_data[];
    
    // extern __constant__ float c_filter[]; // constant memory for the filter fir future implementation
    
    int local_thread_index = threadIdx.x; // Thread of block position index
    int stride = blockDim.x * gridDim.x; // Stride for the loop

    for (int loop = blockDim.x * blockIdx.x; loop < N; loop += stride)
    {
        int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x; // General thread position index
        int shared_index = local_thread_index + (K - 1); // Shared memory index

        if (global_thread_index < N)
            s_data[shared_index] = input[global_thread_index];
        else
            s_data[shared_index] = 0.0f;

        if (local_thread_index < K - 1)
        {
            int halo_index = global_thread_index - (K - 1);
            if (halo_index >= 0)
                s_data[local_thread_index] = input[halo_index];
            else
                s_data[local_thread_index] = 0.0f;
        }
        __syncthreads();

        if (global_thread_index < N)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
                sum += s_data[shared_index - i] * filter[i];
            output[global_thread_index] = sum;
        }
        __syncthreads();
    }
}



int main()
{
    int N = 100000, K = 16;   // number of samples and filter size
    float fs = 1024.0f;    // sampling frequency
    float f = 50.0f;       // signal frequency

    std::vector<float> h_signal(N);
    std::vector<float> h_filter(K);
    std::vector<float> h_filtered_signal(N);

    // R2C output is N/2 + 1
    std::vector<cufftComplex> h_freq(N / 2 + 1);

    // Signal and Noise generation for testing
    for (int n = 0; n < N; n++)
    {
        float t = n / fs; // Time
        float noise = 0.2f * ((float)rand() / RAND_MAX - 0.5f); // Noise generation
        h_signal[n] = sin(2.0f * PI * f * t) + noise; // Add noise to signal
    }

    // Low pass filter
    for (int i = 0; i < K; i++)
    {
        h_filter[i] = 1.0f / K;
    }

    float* d_signal, * d_filter, * d_filtered_signal;
    cufftComplex* d_freq = nullptr;;
    cudaErrorCheck(cudaMalloc((void**)&d_signal, N * sizeof(float)), "d_signal malloc");
    cudaErrorCheck(cudaMalloc((void**)&d_filter, K * sizeof(float)), "d_filter malloc");
    cudaErrorCheck(cudaMalloc((void**)&d_filtered_signal, N * sizeof(float)), "d_output malloc");
    cudaErrorCheck(cudaMalloc((void**)&d_freq, (N / 2 + 1) * sizeof(cufftComplex)), "d_freq malloc");
    cudaErrorCheck(cudaMemcpy(d_signal, h_signal.data(), N * sizeof(float), cudaMemcpyHostToDevice), "h_signal -> d_signal copy");
    cudaErrorCheck(cudaMemcpy(d_filter, h_filter.data(), K * sizeof(float), cudaMemcpyHostToDevice), "h_filter -> d_filter copy");
    
    // Device attributes for better utillization of the GPU
    int device_id;
    cudaGetDevice(&device_id);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id);
    int BLOCKS_PER_GRID = numSMs * 32;
    int shared_mem_size = (THREADS_PER_BLOCK + K - 1) * sizeof(float);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    FilteringKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, shared_mem_size>>>(d_signal, d_filter, d_filtered_signal, N, K);
    cudaErrorCheck(cudaGetLastError(), "Kernel launch");
    cudaErrorCheck(cudaDeviceSynchronize(), "Sync");

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU time: " << gpu_time << " ms\n";

    cudaErrorCheck(cudaMemcpy(h_filtered_signal.data(), d_filtered_signal, N * sizeof(float), cudaMemcpyDeviceToHost), "d_filterd_signal -> h_filterded_signal copy");

    // FFT exec
    cufftHandle plan;
    cufftResult fft_status = cufftPlan1d(&plan, N, CUFFT_R2C, 1);
    if (fft_status != CUFFT_SUCCESS) {
        std::cout << "cufftPlan1d failed\n";
        return 1;
    }
    fft_status = cufftExecR2C(plan, d_filtered_signal, d_freq);
    if (fft_status != CUFFT_SUCCESS) {
        std::cout << "cufftExecR2C failed\n";
        return 1;
    }
    cudaErrorCheck(cudaDeviceSynchronize(), "FFT sync");
    cudaMemcpy(h_freq.data(), d_freq, (N / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Writting vectors to files for storage
    // std::filesystem::create_directory("output_files");
    std::ofstream file_unfiltered("unfiltered_output.txt");
    std::ofstream file_filtered("filtered_output.txt");
    std::ofstream file_frequency("frequency_filtered_output.txt");

    for (int i = 0; i < N; i++)
        file_unfiltered << h_signal[i] << std::endl;

    for (int i = 0; i < N; i++)
        file_filtered << h_filtered_signal[i] << std::endl;

    for (int i = 0; i < N / 2 + 1; i++)
    {
        float real = h_freq[i].x;
        float imag = h_freq[i].y;
        float magnitude = sqrt(real * real + imag * imag);
        float freq = i * fs / N;
        file_frequency << freq << " " << magnitude << std::endl;
    }

    file_filtered.close();
    file_unfiltered.close();
    file_frequency.close();

    // Memory cleanup
    cudaErrorCheck(cudaFree(d_signal), "d_signal free");
    cudaErrorCheck(cudaFree(d_filtered_signal), "d_filtered_signal free");
    cudaErrorCheck(cudaFree(d_filter), "d_filter free");

    return 0;
}