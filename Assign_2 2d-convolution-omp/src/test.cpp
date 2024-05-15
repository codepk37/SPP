// #include <immintrin.h> // Include AVX-512 intrinsics
// #include <iostream>

// // Function to print the contents of an AVX-512 register
// void print_avx512_register(__m512 reg) {
//     // Create an array to store the contents of the register
//     float elements[16];
    
//     // Extract elements from the AVX-512 register and store them in the array
//     _mm512_storeu_ps(elements, reg);
    
//     // Print each element in the array
//     for (int i = 0; i < 16; ++i) {
//         std::cout << elements[i] << " ";
//     }
//     std::cout << std::endl;
// }

// int main() {
//     // Example AVX-512 register containing some data
//     __m512 avx_reg = _mm512_set_ps(1.0, 1.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0,
//                                     9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 15.0, 16.0);
    
//     // Print the contents of the AVX-512 register
//     print_avx512_register(avx_reg);
    
//     return 0;
// }



#include <immintrin.h>
#include <iostream>

float DotProductAVX2(const float* a, const float* b) {
    __m256 vec_a = _mm256_set_ps(a[3], a[2], a[1], a[0],a[4],a[5],0,0);
    __m256 vec_b = _mm256_set_ps(b[3], b[2], b[1], b[0], b[4],b[5],0,0);
    __m256 product = _mm256_mul_ps(vec_a, vec_b);
    
    // Horizontal addition
    __m128 sum_low = _mm256_extractf128_ps(product, 1);
    __m128 sum_high= _mm256_castps256_ps128(product);
    __m128 sum = _mm_add_ps(sum_low, sum_high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    
    // Extract the result
    float result;
    _mm_store_ss(&result, sum);
    return result;
}

int main() {
    float a[6] = {3.0f, 2.0f, 1.0f, 1.0f,2,1};
    float b[6] = {1.0f, 1.0f, 1.0f, 1.0f,1,1};

    float result = DotProductAVX2(a, b);
    std::cout << "Dot Product: " << result << std::endl;

    return 0;
}
