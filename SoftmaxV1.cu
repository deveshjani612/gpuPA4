
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cuda.h>

#define BLOCK_SIZE 256

__global__ void softmax_naive_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float max_val = row_input[0];
    for (int i = 1; i < cols; ++i) {
        if (row_input[i] > max_val) {
            max_val = row_input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        sum += expf(row_input[i] - max_val);
    }

    for (int i = 0; i < cols; ++i) {
        row_output[i] = expf(row_input[i] - max_val) / sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./SoftmaxV1 input.txt" << std::endl;
        return 1;
    }

    std::ifstream infile(argv[1]);
    int rows, cols;
    infile >> rows >> cols;

    int total_elems = rows * cols;
    float* h_input = new float[total_elems];
    float* h_output = new float[total_elems];

    for (int i = 0; i < total_elems; ++i) {
        infile >> h_input[i];
    }
    infile.close();

    float *d_input, *d_output;
    cudaMalloc(&d_input, total_elems * sizeof(float));
    cudaMalloc(&d_output, total_elems * sizeof(float));

    cudaMemcpy(d_input, h_input, total_elems * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int num_blocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_naive_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, total_elems * sizeof(float), cudaMemcpyDeviceToHost);

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
