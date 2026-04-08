#include <iostream>
#include <cuda_runtime.h>
#include "radiometer.cuh"

int main()
{
    const int N = 5;
    float2 input[N];
    float output[N] = { 0 };
    input[0] = { 1.0f, 2.0f };
    input[1] = { 2.0f, 3.0f };
    input[2] = { 3.0f, 4.0f };
    input[3] = { 4.0f, 5.0f };
    input[4] = { 5.0f, 6.0f };

    radiometer(input, output, N);

    for (int i = 0; i < N; i++)
    {
        std::cout << "Sample " << i
                  << ": I = " << input[i].x
                  << ", Q = " << input[i].y
                  << ", Mag^2 = " << output[i] << "\n";
    }

    return 0;
}