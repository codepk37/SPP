
// 160000 elements in block as: L1 786kB
// 400*400 sub block elemnt


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

// Function to read matrix from a text file
std::vector<std::vector<int>> read_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<int>> matrix;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int num;
        std::vector<int> row;
        while (iss >> num) {
            row.push_back(num);
        }
        matrix.push_back(row);
    }

    return matrix;
}

// Function to write matrix to a text file
void write_matrix_to_file(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        return;
    }
    for (const auto& row : matrix) {
        for (int elem : row) {
            file << elem << " ";
        }
        file << std::endl;
    }
}

// CORRECT CODE FOR BLOCKING ,ANY SIZE OF BLOCK
//now do for A*B with outer product

// Function to compute block matrix multiplication
/* B is not transposed
std::vector<std::vector<int>> block_matrix_multiply(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int blockSize) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();
    
    std::vector<std::vector<int>> result(m, std::vector<int>(p, 0));
    
    for (int i = 0; i < m; i += blockSize) {
        for (int j = 0; j < p; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                // Compute block multiplication
                // for ii: i-> min(i + blockSize, m)
                    // for jj: j -> min(j + blockSize, p)
                for (int ii = i; ii < std::min(i + blockSize, m); ++ii) {
                    for (int jj = j; jj < std::min(j + blockSize, p); ++jj) {
                        for (int kk = k; kk < std::min(k + blockSize, n); ++kk) {
                            result[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    
    return result;
}
*/
// Function to compute block matrix multiplication
std::vector<std::vector<int>> block_matrix_multiply(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int blockSize) {
    int m = A.size();
    int n = A[0].size();
    int p = B.size(); // Use the number of rows of B for transpose
    
    // Transpose matrix B : vectorization can be used
    std::vector<std::vector<int>> B_transpose(p, std::vector<int>(n));
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < n; ++j) {
            B_transpose[i][j] = B[j][i];
        }
    }
    
    std::vector<std::vector<int>> result(m, std::vector<int>(p, 0));
    
    for (int i = 0; i < m; i += blockSize) {
        for (int j = 0; j < p; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                // Compute block multiplication

                for (int ii = i; ii < std::min(i + blockSize, m); ++ii) {
                    for (int jj = j; jj < std::min(j + blockSize, p); ++jj) {
                        for (int kk = k; kk < std::min(k + blockSize, n); ++kk) {
                            result[ii][jj] += A[ii][kk] * B_transpose[jj][kk]; // Use transpose of B
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

int main() {
    // Read matrices from files
    std::vector<std::vector<int>> A = read_matrix_from_file("1.txt");
    std::vector<std::vector<int>> B=  read_matrix_from_file("2.txt");
    
    // Block size
    int blockSize = 128; // Modify the block size as needed
    
    // Compute block matrix multiplication
    std::vector<std::vector<int>> result = block_matrix_multiply(A, B, blockSize);
    
    // Write result matrix to file
    write_matrix_to_file(result, "out.txt");
    
    std::cout << "Result matrix has been written to out.txt" << std::endl;
    
    return 0;
}
