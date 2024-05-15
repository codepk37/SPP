#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

// g++ -mavx512f my.cpp
// advixe-cl -collect hotspots -- ./a.out
// python3 checker.py
// advisor-gui ./src.advixeproj

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include<omp.h>
#include<bits/stdc++.h>
using namespace std;
#include <immintrin.h>
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <iostream>
#include <functional>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <exception>
#include <memory>
#include <fstream>
#include <immintrin.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <studentlib.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include<omp.h>
#include<bits/stdc++.h>
using namespace std;
#include <immintrin.h>
// //3050ms binary single write
// //2500ms read mmap
// 2400ms read write mmap
// 2700 old

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace solution {
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
        int fd = open(sol_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR); // Open the file with read-write permissions
        if (fd == -1) {
            std::cerr << "Failed to open file: " << sol_path << std::endl;
            return ""; // Return empty string indicating failure
        }
        
        // Calculate the file size needed to store the floating point values
        off_t file_size = sizeof(float) * num_rows * num_cols;
        if (ftruncate(fd, file_size) == -1) {
            std::cerr << "Failed to resize file: " << sol_path << std::endl;
            close(fd);
            return ""; // Return empty string indicating failure
        }
        
        // Map the file into memory
        void* addr = mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            std::cerr << "Failed to mmap file: " << sol_path << std::endl;
            close(fd);
            return ""; // Return empty string indicating failure
        }
        
        // Perform computation and write directly to memory-mapped file
        float* data = static_cast<float*>(addr);
     
		std::int32_t fd_sol = open(bitmap_path.c_str(), O_RDONLY);
		const auto img = reinterpret_cast<float*>(mmap(nullptr, file_size , PROT_READ, MAP_PRIVATE, fd_sol, 0));
		
		// float* img = new float[num_rows*num_cols];

		// #pragma omp parallel for schedule(dynamic, 100000000) ////####
		// for(int z=0;z<num_cols*num_rows;z++){
		// 	img[z]=img2[z];
		// }



		//SATRT : num_rows;num_cols;img -;kernel -> data[k]
		int r=num_rows,c=num_cols;
		cout<<"                   "<<num_rows<<"  "<<num_cols<<endl;

		// float cop[2][r];  //truing to reuce cache miss, by transposing 
		// float cop2[2][r];  //truing to reuce cache miss, by transposing 
		
		// #pragma omp for collapse(2) schedule(dynamic,10000)  //UNROLLING IN SINGLE LOOP 
		// for(int i=0;i<2;++i){
		// 	for(int j=0;j<r;++j){
		// 		cop[i][j]=img[j*c +i];
		// 		cop2[i][j]=img[j*c +c-1-i];
		// 	}
		// }

		

		int row_chunk=r/5;//3 gave best
		

        #pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			
			// Set thread affinity
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(thread_id, &cpuset);
			if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
				perror("sched_setaffinity");
				// Handle error
			}





			//////////////////////////////////inner convoultion , regardless of cloumn size : multipleof 16 or not , does well 
			#pragma omp single
			for(int i=1;i<r;i+=row_chunk){
				int s=i; int e=min(i+row_chunk-1,r-2);
				// std::cout<<"     "<<s<<"  "<<e<<"   \n";

				#pragma omp task
				{
					_mm_prefetch((const char*)&kernel[0][0], _MM_HINT_T0);
				for (int row = s; row <= e; ++row) {
					
					/////
					//col=0 ,removed loop
					_mm_prefetch((const char*)&kernel[0][0], _MM_HINT_T0);
					int col_0=0;
					data[row * c+col_0]= img[(row-1)*c  +col_0]*kernel[0][1] + img[(row-1)*c  +col_0+1]*kernel[0][2] + img[(row)*c  +col_0]*kernel[1][1] + img[(row)*c  +col_0+1]*kernel[1][2]  +  img[(row+1)*c  +col_0]*kernel[2][1]  + img[(row+1)*c   +col_0+1]*kernel[2][2] ;
					/////



					for (int col = 1; col <= c/2; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
						__m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible

						//
						// if row=1 ,col:1-> c-2
						if(row==1){
							int row_0=0;
									__m512 sum2 = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
									for (int ky = 0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
										for (int kx = -1; kx <= 1; ++kx) {
											__m512 pixels2 = _mm512_loadu_ps(&img[(row_0 + ky)*c  +col + kx]);
											
											__m512 filterVal2 = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
											sum2 = _mm512_fmadd_ps(pixels2, filterVal2, sum2);
										}
									// }
									// _mm512_storeu_ps(&arr[row_0][col], sum);
									_mm512_storeu_ps(&data[row_0*c +col], sum2);
							}
						}
						
						
						//


						// Prefetch kernel values
						

						for (int ky = -1; ky <= 1; ++ky) {
							for (int kx = -1; kx <= 1; ++kx) {
								// printf("%d %d\n",row,col);

								// Prefetch data for the current kernel position
								_mm_prefetch((const char*)&img[(row + ky) * c + col + kx], _MM_HINT_T1);


								// printf("--- %d %d\n",(row + ky) , (col + kx));
								__m512 pixels = _mm512_loadu_ps(&img[(row + ky)*c+ col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
						// _mm512_storeu_ps(&arr[row][col], sum);
						_mm512_storeu_ps(&data[row*c +col], sum);


						if(row==r-2){
							int row_r_1=r-1;
									__m512 sum3 = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
									for (int ky = -1; ky <= 0; ++ky) {  //ky =1 not possible
										for (int kx = -1; kx <= 1; ++kx) {
											__m512 pixels3 = _mm512_loadu_ps(&img[(row_r_1 + ky)*c   +col + kx]);
											
											__m512 filterVal3 = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
											sum3 = _mm512_fmadd_ps(pixels3, filterVal3, sum3);
										}
									}
									// _mm512_storeu_ps(&arr[row_r_1][col], sum);
									_mm512_storeu_ps(&data[row_r_1 *c+col], sum3);
							// }

						}

						
					}
					
					
					}


				}

				#pragma omp task
				{
					_mm_prefetch((const char*)&kernel[0][0], _MM_HINT_T0);
					_mm_prefetch((const char*)&kernel[0][0], _MM_HINT_T1);
				for (int row = s; row <= e; ++row) {
					for (int col = c/2+1; col <= c - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
						__m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible

						///
						_mm_prefetch((const char*)&kernel[0][0], _MM_HINT_T0);
						if(row==1){
							int row_0=0;
									__m512 sum2 = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
									for (int ky = 0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
										for (int kx = -1; kx <= 1; ++kx) {
											__m512 pixels2 = _mm512_loadu_ps(&img[(row_0 + ky)*c  +col + kx]);
											
											__m512 filterVal2 = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
											sum2 = _mm512_fmadd_ps(pixels2, filterVal2, sum2);
										}
									// }
									// _mm512_storeu_ps(&arr[row_0][col], sum);
									_mm512_storeu_ps(&data[row_0*c +col], sum2);
							}
						}
						
						///


						 // Prefetch kernel values
        				

						for (int ky = -1; ky <= 1; ++ky) {
							for (int kx = -1; kx <= 1; ++kx) {
								// printf("%d %d\n",row,col);

								// Prefetch data for the current kernel position
				                _mm_prefetch((const char*)&img[(row + ky) * c + col + kx], _MM_HINT_T1);


								// printf("--- %d %d\n",(row + ky) , (col + kx));
								__m512 pixels = _mm512_loadu_ps(&img[(row + ky)*c+ col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
						// _mm512_storeu_ps(&arr[row][col], sum);
						_mm512_storeu_ps(&data[row*c +col], sum);


						if(row==r-2){
							int row_r_1=r-1;
									__m512 sum3 = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
									for (int ky = -1; ky <= 0; ++ky) {  //ky =1 not possible
										for (int kx = -1; kx <= 1; ++kx) {
											__m512 pixels3 = _mm512_loadu_ps(&img[(row_r_1 + ky)*c   +col + kx]);
											
											__m512 filterVal3 = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
											sum3 = _mm512_fmadd_ps(pixels3, filterVal3, sum3);
										}
									}
									// _mm512_storeu_ps(&arr[row_r_1][col], sum);
									_mm512_storeu_ps(&data[row_r_1 *c+col], sum3);
							// }

						}


					}
					//remaing part <16 
					if((c-2)%16!=0){
						
						int rem=(c-2)%16;
						int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 

						
						__m512 sum = _mm512_setzero_ps(); 

						for (int ky = -1; ky <= 1; ++ky) {
							for (int kx = -1; kx <= 1; ++kx) {
								// printf("%d %d\n",row,remain_single_part_start_col);
								// printf("--- %d %d\n",(row + ky) , (remain_single_part_start_col + kx));
								__m512 pixels = _mm512_loadu_ps(&img[ c*(row + ky) +remain_single_part_start_col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}

						/////
						//row=1-r-2, col=c-1
						int col_c_1=c-1;
						data[row * c+col_c_1]=img[(row-1)*c   +col_c_1-1]*kernel[0][0] + img[(row-1)*c   +col_c_1]*kernel[0][1] + img[(row)*c   +col_c_1-1]*kernel[1][0] + img[(row)*c    +col_c_1]*kernel[1][1]  +  img[(row+1)*c   +col_c_1-1]*kernel[2][0]  + img[(row+1)*c    +col_c_1]*kernel[2][1] ;
						/////


						////row=0 , do it at row=1 ,Only Once
						if(row==1){
							int row_0=0;
									__m512 sum2 = _mm512_setzero_ps(); 

								for (int ky =0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
									for (int kx = -1; kx <= 1; ++kx) {
										
										__m512 pixels2 = _mm512_loadu_ps(&img[(row_0 + ky)*c   +remain_single_part_start_col + kx]);
										
										__m512 filterVal2 = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
										sum2 = _mm512_fmadd_ps(pixels2, filterVal2, sum2);
									}
								}
								float elements2[16];
								_mm512_storeu_ps(elements2, sum2);   
								for(int j=0;j<rem;++j){ //to avoid junk values,also made independent for pragma by not updating other's part not update
									// arr[row_0][remain_single_part_start_col+j]=elements[j];
									data[row_0 *c +remain_single_part_start_col+j]=elements2[j];
									// printf("%d!\n",elements[j]);
								}
							// }
						}
												//////
			


						// _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
						float elements[16];
						_mm512_storeu_ps(elements, sum);   
						for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
							// arr[row][remain_single_part_start_col+j]=elements[j];
							data[row *c +remain_single_part_start_col+j]=elements[j];
							// printf("%d!\n",elements[j]);
						}			


						//only once
						if(row==r-2){//row=r-1  excluding corner
							int row_r_1=r-1;
								
									__m512 sum3 = _mm512_setzero_ps(); 
									

								for (int ky =-1; ky <= 0; ++ky) { //ky =1 not possible
									for (int kx = -1; kx <= 1; ++kx) {
										
										__m512 pixels3 = _mm512_loadu_ps(&img[(row_r_1 + ky)*c  +remain_single_part_start_col + kx]);
										
										__m512 filterVal3 = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
										sum3 = _mm512_fmadd_ps(pixels3, filterVal3, sum3);
									}
								}
								// _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
								float elements3[16];
								_mm512_storeu_ps(elements3, sum3);   
								for(int j=0;j<rem;++j){ //to avoid junk values,also made independent for pragma by not updating other's part not update
									// arr[row_r_1][remain_single_part_start_col+j]=elements[j];
									data[row_r_1 *c +remain_single_part_start_col+j]=elements3[j];
									// printf("%d!\n",elements[j]);
								}
							// }
						}

			

					}
				}
				}


			}

			
			
			
			//////////////////////////////////inner convoultion , regardless of cloumn size : multipleof 16 or not , does well above



			/////////////// only corners 
			#pragma omp single nowait
			{
			data[0*c+0]=  img[0*c+0]*kernel[1][1] + img[0*c+1]*kernel[1][2] + img[1*c+0]*kernel[2][1] + img[1*c+1]*kernel[2][2] ;
			// =arr[0][0];
			data[0*c  +c-1]=  img[0*c+c-2]*kernel[1][0] + img[0*c+  c-1]*kernel[1][1] + img[1*c +c-2]*kernel[2][0] + img[1*c  +c-1]*kernel[2][1] ;
			//  =arr[0][c-1];

			data[ (r-1)*c +0]=  img[(r-2)*c +0]*kernel[0][1] + img[(r-2)*c+1]*kernel[0][2] + img[(r-1)*c  +0]*kernel[1][1] + img[(r-1)*c  +1]*kernel[1][2] ;
			// =arr[r-1][0];
			
			data[(r-1 )*c +c-1] = img[(r-2)*c  +c-2]*kernel[0][0] + img[(r-2)*c  +c-1]*kernel[0][1] + img[(r-1)*c  +c-2]*kernel[1][0] + img[(r-1)*c +c-1]*kernel[1][1] ;
			// =arr[r-1][c-1];
			}
			//////////////

			
			// {
			
			
				// int col_0=0;
				// #pragma omp for schedule(dynamic,10000) nowait
				// for(int i=1;i<=r-2;i++){
				// 	// arr[i][col_0]=  img[(i-1)*c  +col_0]*kernel[0][1] + img[(i-1)*c  +col_0+1]*kernel[0][2] + img[(i)*c  +col_0]*kernel[1][1] + img[(i)*c  +col_0+1]*kernel[1][2]  +  img[(i+1)*c  +col_0]*kernel[2][1]  + img[(i+1)*c   +col_0+1]*kernel[2][2] ;
				// 	data[i * c+col_0]= img[(i-1)*c  +col_0]*kernel[0][1] + img[(i-1)*c  +col_0+1]*kernel[0][2] + img[(i)*c  +col_0]*kernel[1][1] + img[(i)*c  +col_0+1]*kernel[1][2]  +  img[(i+1)*c  +col_0]*kernel[2][1]  + img[(i+1)*c   +col_0+1]*kernel[2][2] ;
				// }
			

			
			
				// int col_c_1=c-1;
				// #pragma omp for schedule(dynamic,10000) nowait
				// for(int i=1;i<=r-2;i++){
				// 	// arr[i][col_c_1]=  img[(i-1)*c   +col_c_1-1]*kernel[0][0] + img[(i-1)*c   +col_c_1]*kernel[0][1] + img[(i)*c   +col_c_1-1]*kernel[1][0] + img[(i)*c    +col_c_1]*kernel[1][1]  +  img[(i+1)*c   +col_c_1-1]*kernel[2][0]  + img[(i+1)*c    +col_c_1]*kernel[2][1] ;
				// 	data[i * c+col_c_1]=img[(i-1)*c   +col_c_1-1]*kernel[0][0] + img[(i-1)*c   +col_c_1]*kernel[0][1] + img[(i)*c   +col_c_1-1]*kernel[1][0] + img[(i)*c    +col_c_1]*kernel[1][1]  +  img[(i+1)*c   +col_c_1-1]*kernel[2][0]  + img[(i+1)*c    +col_c_1]*kernel[2][1] ;
				// }
			
			// }
			


			// auto start1 = std::chrono::high_resolution_clock::now();
			/////row=0 excluding corner
			// #pragma omp single
			/*
			{
				int row_0=0;
				#pragma omp for schedule(dynamic,100)
				for (int col = 1; col <= c - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
						__m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
						for (int ky = 0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
							for (int kx = -1; kx <= 1; ++kx) {
								// printf("%d %d\n",row_0,col);
								// printf("--- %d %d\n",(row_0 + ky) , (col + kx));
								__m512 pixels = _mm512_loadu_ps(&img[(row_0 + ky)*c  +col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
						// _mm512_storeu_ps(&arr[row_0][col], sum);
						_mm512_storeu_ps(&data[row_0*c +col], sum);


				}
				#pragma omp single 
				{	

					if((c-2)%16!=0){
						int rem=(c-2)%16;
						int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 
							__m512 sum = _mm512_setzero_ps(); 

						for (int ky =0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
							for (int kx = -1; kx <= 1; ++kx) {
								
								__m512 pixels = _mm512_loadu_ps(&img[(row_0 + ky)*c   +remain_single_part_start_col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
						// _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
						float elements[16];
						_mm512_storeu_ps(elements, sum);   
						for(int j=0;j<rem;++j){ //to avoid junk values,also made independent for pragma by not updating other's part not update
							// arr[row_0][remain_single_part_start_col+j]=elements[j];
							data[row_0 *c +remain_single_part_start_col+j]=elements[j];
							// printf("%d!\n",elements[j]);
						}
					}
				}


			}*/
			/////row=r-1  excluding corner
			// #pragma omp single
			// #pragma omp task
			/*
			{
				int row_r_1=r-1;
				#pragma omp for schedule(dynamic,1000)
				for (int col = 1; col <= c - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
						__m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
						for (int ky = -1; ky <= 0; ++ky) {  //ky =1 not possible
							for (int kx = -1; kx <= 1; ++kx) {
								// printf("%d %d\n",row_r_1,col);
								// printf("--- %d %d\n",(row_r_1 + ky) , (col + kx));
								__m512 pixels = _mm512_loadu_ps(&img[(row_r_1 + ky)*c   +col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
						// _mm512_storeu_ps(&arr[row_r_1][col], sum);
						_mm512_storeu_ps(&data[row_r_1 *c+col], sum);


				}
				#pragma omp single 
				{
					if((c-2)%16!=0){
						int rem=(c-2)%16;
						int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 
							__m512 sum = _mm512_setzero_ps(); 
							

						for (int ky =-1; ky <= 0; ++ky) { //ky =1 not possible
							for (int kx = -1; kx <= 1; ++kx) {
								
								__m512 pixels = _mm512_loadu_ps(&img[(row_r_1 + ky)*c  +remain_single_part_start_col + kx]);
								
								__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
								sum = _mm512_fmadd_ps(pixels, filterVal, sum);
							}
						}
						// _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
						float elements[16];
						_mm512_storeu_ps(elements, sum);   
						for(int j=0;j<rem;++j){ //to avoid junk values,also made independent for pragma by not updating other's part not update
							// arr[row_r_1][remain_single_part_start_col+j]=elements[j];
							data[row_r_1 *c +remain_single_part_start_col+j]=elements[j];
							// printf("%d!\n",elements[j]);
						}
					}
				}


			}*/

		}
			// auto end1 = std::chrono::high_resolution_clock::now();
			// auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
			// std::cout << "----1 Row Elapsed time: " << duration1.count()/(1000*2) << " ms" << std::endl;


			// auto start2 = std::chrono::high_resolution_clock::now();
			
			// auto end2 = std::chrono::high_resolution_clock::now();
			// auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
			// std::cout << "----Time to transpose 2* clon->row " << duration2.count()/(1000) << " ms" << std::endl;


			//  auto start = std::chrono::high_resolution_clock::now();

			/////// for col=0 , col= c-1 , row: 1->r-1

			//
			//
			//
			
			
			
			
			// float cop[2][r];  //truing to reuce cache miss, by transposing 
			// float cop2[2][r];  //truing to reuce cache miss, by transposing 
			// #pragma omp for collapse(2) schedule(dynamic:100000)  //UNROLLING IN SINGLE LOOP 
			// for(int i=0;i<2;++i){
			// 	for(int j=0;j<r;++j){
			// 		cop[i][j]=img[j*c +i];
			// 		cop2[i][j]=img[j*c +c-1-i];
			// 	}
			// } DECLARED ABOVE PARALLEL
			
			
			//

/*	
	{   /////// for col=0 ,  , row: 1->r-1				

			int col_0=0;
			// for(int i=1;i<=r-2;i++){
			//     arr[i][col_0]=  cop[col_0][i-1]*kernel[0][1] + cop[col_0+1][i-1]*kernel[0][2] + cop[col_0][i]*kernel[1][1] + cop[col_0+1][i]*kernel[1][2]  +  cop[col_0][i+1]*kernel[2][1]  + cop[col_0+1][i+1]*kernel[2][2] ;
			// }

			

			int row_0=0; //its col_0
			
			float *temp=new float[r];
			#pragma omp parallel for schedule(dynamic,10000) 
			for(int k=0;k<r;++k){temp[k]=0;}

			#pragma omp for  schedule(dynamic,1000) nowait
			for (int col = 1; col <= r - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
					__m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible


					for (int ky = 0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
						for (int kx = -1; kx <= 1; ++kx) {
							// printf("%d %d\n",row_0,col);
							// printf("--- %d %d\n",(row_0 + ky) , (col + kx));
							// printf("dedesrdesrdesrd %d  %d  \n",row_0,col);
							__m512 pixels = _mm512_loadu_ps(&cop[ky][ col - kx]);                        
							__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);

							// printf("%d %d     %d %d\n",ky,col-kx,   kx+1,ky+1);

							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
					// _mm512_storeu_ps(&arr[row_0][col], sum);                
					// _mm512_storeu_ps(&temp[0], sum);
					// printf("\n        %d        %d\n",r,col);
					_mm512_storeu_ps(&temp[col], sum);

					

			}
			// #pragma omp single nowait
			{
				if((r-2)%16!=0){
					int rem=(r-2)%16;
					
					int remain_single_part_start_col=((int)(r-2)/16)*16+1;//1+ for dest matrix posn 
					// printf("ermewr rem %d     %d\n",rem,remain_single_part_start_col);
						__m512 sum = _mm512_setzero_ps(); 

					for (int ky =0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
						for (int kx = -1; kx <= 1; ++kx) {                        
							__m512 pixels = _mm512_loadu_ps(&cop[(row_0 + ky)][remain_single_part_start_col - kx]);
							
							__m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
							sum = _mm512_fmadd_ps(pixels, filterVal, sum);
						}
					}
					// _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
					float elements[16];
					_mm512_storeu_ps(elements, sum);   
					for(int j=0;j<rem;++j){ //to avoid junk values,also made independent for pragma by not updating other's part not update
						// arr[row_0][remain_single_part_start_col+j]=elements[j];
						// data[row_0 *r +remain_single_part_start_col+j]=elements[j];
						temp[remain_single_part_start_col+j] =elements[j];
						// printf("%d!\n",elements[j]);
					}
				}
			}

			#pragma omp for schedule(dynamic,10000) nowait
			for(int jk = 1; jk <= r - 2; jk++) {
				data[jk * c + 0] = temp[jk];
				// Unroll loop by 2
				if (jk < r - 2) {
					jk++;
					data[jk * c + 0] = temp[jk];
				}
			}
			
		}

		{   
			//col = c-1
			

			int col_c_1=0;
		

			//
			int row_0=0; //its col_0
			float *temp=new float[r];
			// #pragma omp  parallel for  schedule(dynamic,1000) 
			for(int k=0;k<r;++k){temp[k]=0;}
			// #pragma omp parallel for schedule(dynamic,100) 
			for (int col = 1; col <= r - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
					__m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible

					{
					__m512 pixels = _mm512_loadu_ps(&cop2[1][ col-1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[0 ][0 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}{
					__m512 pixels = _mm512_loadu_ps(&cop2[1][ col]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[1 ][0 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}{
					__m512 pixels = _mm512_loadu_ps(&cop2[1][ col+1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[2 ][0 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}{
					__m512 pixels = _mm512_loadu_ps(&cop2[0][ col-1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[0 ][1 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
					__m512 pixels = _mm512_loadu_ps(&cop2[0][ col]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[1 ][1 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					{
					__m512 pixels = _mm512_loadu_ps(&cop2[0][ col+1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[2 ][1 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}




					// _mm512_storeu_ps(&arr[row_0][col], sum);                
					// _mm512_storeu_ps(&temp[0], sum);
					// printf("\n        %d        %d\n",r,col);
					_mm512_storeu_ps(&temp[col], sum);

					

			}
			// #pragma omp single nowait
			{
				if((r-2)%16!=0){
					int rem=(r-2)%16;
					
					
					int remain_single_part_start_col=((int)(r-2)/16)*16+1;//1+ for dest matrix posn 
					// printf("ermewr rem %d     %d\n",rem,remain_single_part_start_col);
						__m512 sum = _mm512_setzero_ps(); 

				

					{
					__m512 pixels = _mm512_loadu_ps(&cop2[1][ remain_single_part_start_col-1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[0 ][0 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}{
					__m512 pixels = _mm512_loadu_ps(&cop2[1][ remain_single_part_start_col]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[1 ][0 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}{
					__m512 pixels = _mm512_loadu_ps(&cop2[1][ remain_single_part_start_col+1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[2 ][0 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}{
					__m512 pixels = _mm512_loadu_ps(&cop2[0][ remain_single_part_start_col-1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[0 ][1 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}
					__m512 pixels = _mm512_loadu_ps(&cop2[0][ remain_single_part_start_col]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[1 ][1 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					{
					__m512 pixels = _mm512_loadu_ps(&cop2[0][ remain_single_part_start_col+1]);                        
					__m512 filterVal = _mm512_set1_ps(kernel[2 ][1 ]);
					sum = _mm512_fmadd_ps(pixels, filterVal, sum);
					}

					// _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
					float elements[16];
					_mm512_storeu_ps(elements, sum);   
					for(int j=0;j<rem;++j){ //to avoid junk values,also made independent for pragma by not updating other's part not update
						// arr[row_0][remain_single_part_start_col+j]=elements[j];
						// data[row_0 *r +remain_single_part_start_col+j]=elements[j];
						temp[remain_single_part_start_col+j] =elements[j];
						// printf("%d!\n",elements[j]);
					}
				}
			}





			#pragma omp  parallel for schedule(dynamic,1000)
			for(int jk=1;jk<=r-2;++jk){
				data[jk*c +c-1]=temp[jk];
			}        

			
		}


*/

		///////////////////////////////////////////////////////////////////////////////////////////////

        
        // Unmap the file and close file descriptor
        // munmap(addr, file_size);
        // close(fd);

		// munmap(img, file_size );//remove img mapping
        
        return sol_path;
    }
}; 