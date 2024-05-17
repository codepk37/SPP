#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <omp.h>
#include <vector>

using namespace std;

// 2nd Commented code below this *** is more optimized : 3 times sppedup even than this

// 1 code :170 mili sec

vector<vector<float>> multiply(const vector<vector<float>> &A, const vector<vector<float>> &B)
{

    int block = A.size();
    vector<vector<float>> C(block, vector<float>(block, 0)); // a is already transposed

#pragma omp parallel for
    for (int line = 0; line < block; line++)
    { // submatrix mult using Outer product
        for (int i = 0; i < block; i++)
        {

            for (int j = 0; j < block; j++)
            {
                C[i][j] += A[line][i] * B[line][j];
            }
        }
    }

    return C;
}

vector<vector<float>> get(const vector<vector<float>> &a, int s, int k, int block)
{
    int n = a.size();
    int m = a[0].size();
    vector<vector<float>> temp(block, vector<float>(block)); // Initialize temp with size 64x64

    // Adjust starting position if it's out of bounds
    s = max(0, min(s, n - block));
    k = max(0, min(k, m - block));

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < block; i++)
    {
        for (int j = 0; j < block; j++)
        {
            temp[i][j] = a[s + i][k + j];
        }
    }
    return temp;
}

#include <immintrin.h> // Include the AVX-512 header

std::vector<std::vector<float>> addMatrices(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B)
{
    int n = A.size();
    int m = A[0].size();
    std::vector<std::vector<float>> C(n, std::vector<float>(m)); // 64, 64

// Perform vectorized addition for each element
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; j += 16)
        {                                         // Process 16 elements at a time
            __m512 a = _mm512_loadu_ps(&A[i][j]); // Load elements from A as 512-bit vector
            __m512 b = _mm512_loadu_ps(&B[i][j]); // Load elements from B as 512-bit vector
            __m512 c = _mm512_add_ps(a, b);       // Vectorized addition
            _mm512_storeu_ps(&C[i][j], c);        // Store result to C
        }
    }
    return C;
}

void copyindest(const vector<vector<float>> &Cij, int i, int j, int block, vector<vector<float>> &dest)
{
#pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < block; x++)
    {
        for (int y = 0; y < block; y++)
        {
            dest[i + x][j + y] = Cij[x][y];
        }
    }
}

void matrixMultiply_byblocks(vector<vector<float>> &A, int size, vector<vector<float>> &B, vector<vector<float>> &dest, int block)
{

#pragma omp parallel
    {
        // #pragma omp single
        {
            int divide = 16; // i.e.big matrix represented as samll chunks
#pragma omp for collapse(2) schedule(dynamic)
            for (int xfac = 0; xfac < divide; xfac++)
            {
                for (int yfac = 0; yfac < divide; yfac++)
                {

                    // #pragma omp task
                    for (int i = (xfac)*size / divide; i < (xfac + 1) * size / divide; i += block)
                    { // 64 for now
                        for (int j = (yfac)*size / divide; j < (yfac + 1) * size / divide; j += block)
                        {

                            vector<vector<float>> Cij(block, vector<float>(block, 0));

                            for (int k = 0; k < size; k += block)
                            {

                                // C[i][j] += a[i][k] * b[j][k];

                                // Cij+= get(A,i,k) * get(B,j,k);

                                Cij = addMatrices(Cij, multiply(get(A, k, i, block), get(B, k, j, block))); // send block later
                            }

                            //////independend

                            copyindest(Cij, i, j, block, dest);
                        }
                    }
                }
            }
        }
    }
}

namespace solution
{
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
    {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        // std::ofstream sol_fs(sol_path, std::ios::binary);
        int fd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR); // Open the file with read-write permissions

        off_t file_size = sizeof(float) * n * m;
        if (ftruncate(fd, file_size) == -1)
        {
            std::cerr << "Failed to resize file: " << sol_path << std::endl;
            close(fd);
            return ""; // Return empty string indicating failure
        }
        // Map the file into memory
        void *addr = mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
        float *result = static_cast<float *>(addr);

        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);
        std::vector<float> m1(n * k);
        std::vector<std::vector<float>> m2(k, std::vector<float>(m));

        // Read data from m1_fs into m1
        m1_fs.read(reinterpret_cast<char *>(m1.data()), sizeof(float) * n * k);

        // Read data from m2_fs into m2
        for (int i = 0; i < k; ++i)
        {
            m2_fs.read(reinterpret_cast<char *>(m2[i].data()), sizeof(float) * m);
        }

        m1_fs.close();
        m2_fs.close();

        std::vector<std::vector<float>> m1_tra(k, std::vector<float>(n, 0)); // std::vector<std::vector<float>> m2_tra(k, std::vector<float>(n, 0));
        std::vector<std::vector<float>> m2_m(k, std::vector<float>(m, 0));

        // for(int i=0;i<k;i++)
        //     for(int j=0;j<m;j++)
        //         m2_m[i][j]= m2[i][ j];

        int block_size = 64;
        int rows = n;
        int cols = k;
        // k*m

        for (int i = 0; i < rows; i += block_size)
        {
            for (int j = 0; j < cols; j += block_size)
            {
                int block_rows = std::min(block_size, rows - i);
                int block_cols = std::min(block_size, cols - j);
                for (int a = 0; a < block_rows; a++)
                {
                    for (int b = 0; b < block_cols; b++)
                    {
                        m1_tra[j + b][i + a] = m1[(i + a) * m + j + b];
                        // m2_tra[j + b][i + a] = m2[(i + a)*m+j + b];
                    }
                }
            }
        }

        int block = 64; // divide till
        std::vector<std::vector<float>> result_(n, std::vector<float>(m, 0));
        matrixMultiply_byblocks(m1_tra, m1_tra.size(), m2, result_, block);

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                result[i * m + j] = result_[i][j];
            }
        }

        return sol_path;
    }
};


// 2nd Code : 52 milli sec

// REPORT:
// more optimized :since once submatrix(working set) fits in cache ,iterating columnwise wont make difference
//  above code used making vectors, function calls which make it 3 times slower
// observed 2 times sppedup in avx512 compared to avx2 in below code

// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <algorithm>
// #include <omp.h>
// #include <immintrin.h> // Include the AVX-512 header

// namespace solution {

// std::string compute(const std::string& m1_path, const std::string& m2_path, int n, int k, int m) {
//     std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
//     int fd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR); // Open the file with read-write permissions

//     off_t file_size = sizeof(float) * n*m;
//     if (ftruncate(fd, file_size) == -1) {
//         std::cerr << "Failed to resize file: " << sol_path << std::endl;
//         close(fd);
//         return ""; // Return empty string indicating failure
//     }
//     // Map the file into memory
//     void* addr = mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
//     float* result = static_cast<float*>(addr);

//     // Open input files
//     int m1_fd = open(m1_path.c_str(), O_RDONLY);

//     int m2_fd = open(m2_path.c_str(), O_RDONLY);

//     // Map input and output files
//     float* m1 = (float*)mmap(nullptr, sizeof(float) * n * k, PROT_READ, MAP_PRIVATE, m1_fd, 0);

//     float* m2 = (float*)mmap(nullptr, sizeof(float) * k * m, PROT_READ, MAP_PRIVATE, m2_fd, 0);

//     close(m1_fd);
//     close(m2_fd);

//     int block_size = 128;
//     // Compute the result using AVX-512 instructions
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int row = 0; row < n; row += block_size) {
//         for (int col = 0; col < m; col += block_size) {
//             for (int z = 0; z < k; z += block_size) {

//                 for (int i = row; i < std::min(row + block_size, n); ++i) {
//                     for (int j = col; j < std::min(col + block_size, m); j += 16) {
//                         __m512 accum = _mm512_load_ps(&result[i * m + j]);
//                         for (int p = z; p < std::min(z + block_size, k); ++p) {
//                             __m512 v1 = _mm512_broadcastss_ps(_mm_load_ss(&m1[i * k + p]));
//                             __m512 v2 = _mm512_loadu_ps(&m2[p * m + j]);
//                             accum = _mm512_fmadd_ps(v1, v2, accum);
//                         }
//                         _mm512_storeu_ps(&result[i * m + j], accum);
//                     }
//                 }
//             }
//         }
//     }

//     // Unmap the memory-mapped files

//     return sol_path;
// }

// };
