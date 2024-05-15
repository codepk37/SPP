
#include <iostream>
#include <vector>
#include <omp.h>

// Function to compute the outer product of two vectors
std::vector<std::vector<int>> outer_product(const std::vector<int>& v1, const std::vector<int>& v2) {
    int n = v1.size();
    int m = v2.size();
    
    std::vector<std::vector<int>> result(n, std::vector<int>(m, 0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[i][j] = v1[i] * v2[j];
        }
    }
    
    return result;
}

int main() {
    // Example vectors
    std::vector<int> v1 = {2, 2,1,2,2,2,3}; //Ci
    std::vector<int> v2 = {1, 1,0,1,1,4,5};  //Ri
    
    // Compute outer product
    std::vector<std::vector<int>> result = outer_product(v1, v2);
    
    // Print result
    std::cout << "Outer product:" << std::endl;
    for (const auto& row : result) {
        for (int elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
