#include <iostream>
#include <vector>
#include <immintrin.h> // AVX-512 header
using namespace std;
int main() {
    // Example matrices A, B, C, D
    std::vector<std::vector<std::vector<int>>> matrices = {
        {{1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1  },
         {3, 2, 1, 2, 2, 1, 3, 1, 2, 1, 3, 1, 2, 2, 3, 3,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 }},
         
        {{3, 1, 1, 3, 2, 2, 3, 1, 2, 1, 3, 1, 1, 3, 3, 2,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 },
         {2, 3, 2, 1, 1, 2, 3, 1, 3, 1, 3, 2, 1, 2, 3, 2,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 }},
        
        {{2, 3, 2, 1, 1, 3, 2, 2, 1, 3, 3, 1, 2, 2, 1, 1,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 },
         {1, 3, 1, 2, 2, 1, 3, 1, 2, 1, 3, 3, 2, 1, 3, 1,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 }},
        
        {{3, 1, 1, 3, 2, 2, 3, 1, 2, 1, 3, 1, 1, 3, 3, 2,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 },
         {2, 3, 2, 1, 1, 2, 3, 1, 3, 1, 3, 2, 1, 2, 3, 2,1,1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 3, 1, 3, 2, 1, 3 ,1 }}
    };

    // Determine dimensions of the matrices
    size_t num_matrices = matrices.size();
    size_t num_rows = matrices[0].size();
    size_t num_cols = matrices[0][0].size();
    cout<<"---"<<num_cols<<endl;


    // Result matrix
    std::vector<std::vector<int>> result(num_rows, std::vector<int>(num_cols, 0));

    // Iterate through each matrix in the vector
    for (const auto& matrix : matrices) {
        // Iterate through each row of the matrix
        for (size_t i = 0; i < num_rows; ++i) {
            // Pointers to matrices and result rows
            const int* ptr_matrix = matrix[i].data();
            int* ptr_result = result[i].data();

            // Process 16 elements at a time using AVX-512
            int j=0;
            for (j = 0; j +16 < num_cols; j += 16) {
                // Load 16 elements from the matrix
                __m512i vec_matrix = _mm512_loadu_si512((__m512i*)(ptr_matrix + j));

                // Load current result from the result matrix
                __m512i vec_result = _mm512_loadu_si512((__m512i*)(ptr_result + j));

                // Perform vectorized addition
                vec_result = _mm512_add_epi32(vec_result, vec_matrix);

                // Store the result back to the result matrix
                _mm512_storeu_si512((__m512i*)(ptr_result + j), vec_result);
            }
            
        }
    }

    // Print result
    std::cout << "Result of matrix addition using AVX-512:" << std::endl;
    for (const auto& row : result) {
        for (int elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
