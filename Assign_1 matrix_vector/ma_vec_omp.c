#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // Include SIMD Intrinsics for AVX-512

#define MATRIX_SIZE 10
#define AVX512_SIZE 8  // Number of elements processed in one AVX-512 operation (8 for AVX-512)

double matrix[MATRIX_SIZE][MATRIX_SIZE];
double vector[MATRIX_SIZE];
double result[MATRIX_SIZE];

void initialize() {
    srand(time(NULL));
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = (double)i;//(double)rand() / RAND_MAX;
        }
        vector[i] = 1;//(double)rand() / RAND_MAX;
    }
}

void matrix_vector_multiply() {
    // Perform matrix-vector multiplication
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid == 0) {
            int num_threads = omp_get_num_threads();
            printf("Number of threads: %d\n", num_threads);
        }

        #pragma omp for
        for (int i = 0; i < MATRIX_SIZE; i++) {
            result[i] = 0.0;
            for (int j = 0; j < MATRIX_SIZE; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
    }
}

int main() {
    initialize();

    double start_time = omp_get_wtime();
    matrix_vector_multiply();
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    // Print time taken
    printf("Time taken: %f seconds\n", elapsed_time);

    // Calculate total floating point operations
    long long flops = 2 * MATRIX_SIZE * MATRIX_SIZE;

    // Calculate GFLOPS
    double gflops = flops / (elapsed_time * 1e9);
    printf("GFLOPS: %f\n", gflops);

    printf("\n");
    for(int i=0;i<MATRIX_SIZE;i++){
        printf("%f ",result[i]);   
    }

    return 0;
}
