
// 160000 elements in block as: L1 786kB
// 400*400 sub block elemnt


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <unistd.h> 
#include <fcntl.h>
#include <immintrin.h>
#include <string>
#include <vector>
using namespace std;

// Function to read matrix from a text file
vector<vector<float>> read_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> matrix;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float num;
        std::vector<float> row;
        while (iss >> num) {
            row.push_back(num);
        }
        matrix.push_back(row);
    }

    return matrix;
}

// Function to write matrix to a text file
#include <iostream>
#include <fstream>
#include <vector>

// Function to write matrix to a text file
void write_matrix_to_file(const std::vector<std::vector<float>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        return;
    }
    // Set float formatting to ensure proper precision
    file.precision(8); // Adjust precision as needed

    for (const auto& row : matrix) {
        for (float elem : row) {
            file << elem << " ";
        }
        file << std::endl;
    }
}





vector<vector<float>>  multiply(const vector<vector<float>>& A, const vector<vector<float>>& B) {

    int block= A.size();
    vector<vector<float>> C(block, vector<float>(block, 0)); //a is already transposed

    for(int line=0;line<block;line++){ //submatrix mult using Outer product
            
        for(int i=0;i<block;i++){
            for(int j=0;j<block;j++){
                // #pragma omp atomic
                C[i][j] += A[line][i]* B[line][j];
            }
        }
    }

    // for (int i = 0; i < 64; i++)
	//         for (int j = 0; j < 64; j++) {
	//             C[i][ j] = 0;
	//             for (int l = 0; l < 64; ++l) 
	//             	C[i][j] += A[i][l] * B[l][j];
	//         }
    
    return C;

}


vector<vector<float>> get(const vector<vector<float>>& a, int s, int k,int block) {
    int n = a.size();
    int m = a[0].size();
    vector<vector<float>> temp(block, vector<float>(block)); // Initialize temp with size 64x64

    // Adjust starting position if it's out of bounds
    s = max(0, min(s, n - block));
    k = max(0, min(k, m - block));

    for(int i = 0; i <block; i++) {
        for(int j = 0; j < block; j++) {
            temp[i][j] = a[s + i][k + j];
        }
    }
    return temp;
}



#include <immintrin.h> // Include the AVX-512 header

std::vector<std::vector<float>> addMatrices(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    int n = A.size();
    int m = A[0].size();
    std::vector<std::vector<float>> C(n, std::vector<float>(m)); // 64, 64

    // Perform vectorized addition for each element
    for(int i = 0; i < n; ++i) {
        for (int j = 0; j < m; j += 16) { // Process 16 elements at a time
            __m512 a = _mm512_loadu_ps(&A[i][j]); // Load elements from A as 512-bit vector
            __m512 b = _mm512_loadu_ps(&B[i][j]); // Load elements from B as 512-bit vector
            __m512 c = _mm512_add_ps(a, b);       // Vectorized addition
            _mm512_storeu_ps(&C[i][j], c);        // Store result to C
        }
    }
    return C;
}


void copyindest(const vector<vector<float>> &Cij, int i, int j,int block, vector<vector<float>> &dest) {
    for (int x = 0; x < block; x++) {
        for (int y = 0; y < block; y++) {
            dest[i + x][j + y] = Cij[x][y];
        }
    }
}



void  matrixMultiply_byblocks(vector<vector<float>> &A,int size,vector<vector<float>> &B,vector<vector<float>> &dest,int block){
   
    for (int i = 0; i < size; i += block) {//64 for now
        for (int j = 0; j < size; j += block) {

            vector<vector<float>> Cij(block, vector<float>(block, 0)); 
            
            for (int k=0;k < size ;k+=block){

                // C[i][j] += a[i][k] * b[j][k];

                // Cij+= get(A,i,k) * get(B,j,k);

                Cij=addMatrices(Cij, multiply(get(A,k,i,block), get(B,k,j,block))); //send block later

            }

            //////independend

            copyindest(Cij,i,j,block, dest);

        }
    } 
}




#include <vector>
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
#include <chrono>

int main() {
    using Clock = std::chrono::high_resolution_clock;
    // Read matrices from files
    std::vector<std::vector<float>> m1 = read_matrix_from_file("1.txt");
    std::vector<std::vector<float>> m2=  read_matrix_from_file("2.txt");
    
    // Block size
    int blockSize = 16; // Modify the block size as needed
    
    // Compute block matrix multiplication
    int n = m1.size();
    int k = m1[0].size();
    int m = m2[0].size(); 
    std::vector<std::vector<float>> m2_tra(k, std::vector<float>(m, 0));std::vector<std::vector<float>> m1_tra(k, std::vector<float>(m, 0));
    std::vector<std::vector<float>> result(n, std::vector<float>(m, 0));

    

  
    // //k*m
    const int block_size = 128;
    int rows = k;
    int cols = m;
    //k*m
    for (int i = 0; i < k; i += block_size) {
        for (int j = 0; j < m; j += block_size) {
            int block_rows = std::min(block_size, rows - i);
            int block_cols = std::min(block_size, cols - j);
            for (int a = 0; a < block_rows; a++) {
                for (int b = 0; b < block_cols; b++) {
                    m2_tra[j + b][i + a] = m2[(i + a)][j + b];
                    m1_tra[j + b][i + a] = m1[(i + a)][j + b];
                }
            }
        }
    }
    

    // for (int i = 0; i < n; i++)
	//         for (int j = 0; j < m; j++) {
	//             result[i][ j] = 0;
	//             for (int l = 0; l < k; ++l) 
	//             	result[i][j] += m1[i][l] * m2_tra[j][l];
	//         }


    auto start = Clock::now();
    int block=128; //divide till
    matrixMultiply_byblocks(m1_tra,m2.size(),m2,result,block);



    // for(int line=0;line<k;line++){
            
    //         for(int i=0;i<n;i++){
    //             for(int j=0;j<m;j++){
    //                 // #pragma omp atomic
    //                 result[i][j] += m1_tra[line][i]* m2[line][j];
    //             }
    //         }
    //     }
    auto end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Output the duration
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    // Write result matrix to file
    write_matrix_to_file(result, "out.txt");
    
    std::cout << "Result matrix has been written to out.txt" << std::endl;
    
    return 0;
}