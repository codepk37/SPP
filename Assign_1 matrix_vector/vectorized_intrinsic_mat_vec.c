#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // Include SIMD Intrinsics for AVX-512

#define MATRIX_SIZE 1000
#define AVX512_SIZE 8  // Number of elements processed in one AVX-512 operation (8 for AVX-512)

double matrix[MATRIX_SIZE][MATRIX_SIZE];
double vector[MATRIX_SIZE];
double result[MATRIX_SIZE];

void initialize() {
    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;//i
            printf("%f\n",matrix[i][j]);
        }
        vector[i] = (double)rand() / RAND_MAX;//1
    }
}

void matrix_vector_multiply() {
    // Perform matrix-vector multiplication using manual vectorization with AVX-512
    for (int i = 0; i < MATRIX_SIZE; i++) {
        double sum=0;
        __m512d acc = _mm512_setzero_pd();  // Initialize accumulator with zeros
        int j=0;
        for ( j = 0; j +AVX512_SIZE< MATRIX_SIZE; j += AVX512_SIZE) {
            // Load 8 elements from the vector into AVX-512 register
            __m512d vec = _mm512_loadu_pd(&vector[j]);
            // Load 8 elements from the matrix row into AVX-512 register
            __m512d mat = _mm512_loadu_pd(&matrix[i][j]);
            // Multiply vector and matrix elements
            __m512d prod = _mm512_mul_pd(mat, vec);
            //
            // Add the results to the accumulator
            acc = _mm512_add_pd(acc, prod);
            // sum += _mm512_reduce_add_pd(acc);
            // acc = _mm512_setzero_pd(); 
        }
        sum = _mm512_reduce_add_pd(acc);
        printf("_____j %d       sum %f\n\n",j,sum);
        if(j<MATRIX_SIZE){
            for(j;j<MATRIX_SIZE;j++){
                sum+= matrix[i][j]* vector[j];
            }
        }
        // Store the accumulated result back to the result array
        // _mm512_storeu_pd(&result[i], acc);
        result[i]=sum;
    }
}

int main() {
    initialize();
     for(int i=0;i<MATRIX_SIZE;i++){
        for(int j=0;j<MATRIX_SIZE;j++){
            printf("%f ",matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i=0;i<MATRIX_SIZE;i++){
        printf("%f ",vector[i]);    
    }
    
    

    clock_t start_time = clock();
    matrix_vector_multiply();
    clock_t end_time = clock();
    double elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

   


    // Print time taken
    printf("Time taken: %f seconds\n", elapsed_time);

    // Calculate total floating point operations
    long long flops = 2 * MATRIX_SIZE * MATRIX_SIZE;

    // Calculate GFLOPS
    double gflops = flops / (elapsed_time * 1e9);
    printf("GFLOPS: %f\n", gflops);

    // printf("\n");
    // for(int i=0;i<MATRIX_SIZE;i++){
    //     printf("%f ",result[i]);   
    // }

    return 0;
}

//gcc -mavx512f  vectorized_intrinsic_mat_vec.c 
