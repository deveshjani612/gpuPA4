
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cuda.h>
#include <chrono>

#define BLOCK_SIZE 1024  // assumes cols <= 1024

__global__ void softmax_optimized_kernel(float* input, float* output, int rows, int cols) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= rows || tid >= cols) return;

    float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    shared[tid] = (tid < cols) ? row_input[tid] : -INFINITY;
    __syncthreads();

    // Step 1: Parallel reduction for max
    float max_val = -INFINITY;
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        float temp = __shfl_xor_sync(0xffffffff, shared[tid], stride);
        if (tid % (2 * stride) == 0 && temp > shared[tid]) {
            shared[tid] = temp;
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    float exp_val = expf(row_input[tid] - max_val);
    shared[tid] = exp_val;
    __syncthreads();

    // Step 2: Parallel reduction for sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < cols) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum = shared[0];
    __syncthreads();

    if (tid < cols) {
        row_output[tid] = exp_val / sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./SoftmaxV2 input.txt" << std::endl;
        return 1;
    }

    std::ifstream infile(argv[1]);
    int rows, cols;
    infile >> rows >> cols;

    int total = rows * cols;
    float* h_input = new float[total];
    float* h_output = new float[total];

    for (int i = 0; i < total; ++i) {
        infile >> h_input[i];
    }
    infile.close();

    float *d_input, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));
    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    softmax_optimized_kernel<<<rows, cols, cols * sizeof(float)>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "No of rows: " << rows << std::endl;
    std::cout << "No of cols: " << cols << std::endl;

    for (int i = 0; i < rows; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(3) << h_output[i * cols + j];
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
