Used SIMD AVX512 instruction set for vectorizing code, openmp for using multiple threads, prefetching ,locality of reference ,thread pinning, loop unrolling , fused multiply-add , blocking , mmap in first 3 assignments.<br />
Last assignment was done using cuda <br /><br />

Matrix vector : Optimizations in report <br />
2D image convolution: Optimized till 1.7 sec. Roofline analysis(Intel advisor) gave 1.5 sec <br />
Matrix Matrix Multiplication :Optimized from 10 second(Naive:A * B transpose ) to 52 millisecond's <br />
Cuda convolution : 6777 milli sec on 4096*4096 matrix, 3*/3 kernel
