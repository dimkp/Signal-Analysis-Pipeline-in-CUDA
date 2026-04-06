#include <iostream>
#include "radiometer.cuh"

int main()
{
    const int N = 10;

    float input[N] = {1,2,3,4,5,6,7,8,9,10};
    float output[N] = {0};

    compute_mag2(input, output, N);

    for (int i = 0; i < N; i++)
        std::cout << output[i] << " ";
    
    return 0;
}