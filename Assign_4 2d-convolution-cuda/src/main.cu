
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

namespace solution {
    #define CUDA_ERROR_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); } 

    inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
        if (code != cudaSuccess) {
            fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    __constant__ float kernel[3][3] = {
        { 0.0625f, 0.125f, 0.0625f },
        { 0.125f, 0.25f, 0.125f },
        { 0.0625f, 0.125f, 0.0625f }
    }; // Define kernel as constant memory

    __global__ void convolution2D(const float* img, float* output, int num_rows, int num_cols) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Calculate global indices
        int row = by * blockDim.y + ty;
        int col = bx * blockDim.x + tx;

        // Perform convolution
        if (row < num_rows && col < num_cols) {
            float sum = 0.0f;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int row_idx = row + i;
                    int col_idx = col + j;
                    if (row_idx >= 0 && row_idx < num_rows && col_idx >= 0 && col_idx < num_cols) {
                        sum += kernel[i + 1][j + 1] * img[row_idx * num_cols + col_idx];
                    }
                }
            }
            output[row * num_cols + col] = sum;
        }
    }

    std::string compute(const std::string& bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
        const auto img = std::make_unique<float[]>(num_rows * num_cols);
        bitmap_fs.read(reinterpret_cast<char*>(img.get()), sizeof(float) * num_rows * num_cols);
        bitmap_fs.close();

        // Allocate memory on the GPU
        float* d_img;
        float* d_output;
        CUDA_ERROR_CHECK(cudaMalloc(&d_img, sizeof(float) * num_rows * num_cols));
        CUDA_ERROR_CHECK(cudaMalloc(&d_output, sizeof(float) * num_rows * num_cols));

        // Transfer data to GPU
        CUDA_ERROR_CHECK(cudaMemcpy(d_img, img.get(), sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(32, 32); // Adjust block size as needed
        dim3 gridDim((num_cols + blockDim.x - 1) / blockDim.x, (num_rows + blockDim.y - 1) / blockDim.y);

        // Call CUDA kernel
        convolution2D<<<gridDim, blockDim>>>(d_img, d_output, num_rows, num_cols);
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        // Transfer result back to CPU
        float* output = new float[num_rows * num_cols];
        CUDA_ERROR_CHECK(cudaMemcpy(output, d_output, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

        // Write output to file
        sol_fs.write(reinterpret_cast<char*>(output), sizeof(float) * num_rows * num_cols);

        // Cleanup
        CUDA_ERROR_CHECK(cudaFree(d_img));
        CUDA_ERROR_CHECK(cudaFree(d_output));
        delete[] output;

        sol_fs.close();
        return sol_path;
    }
}
