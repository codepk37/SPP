#include<omp.h>
#include<bits/stdc++.h>
using namespace std;
#include <immintrin.h>
void disp(vector<vector<int>> m){
    for(auto it: m){
        for(auto i : it){
            std::cout<<i<<" ";
        }std::cout<<endl;
    }
}

std::vector<std::vector<int>> readMatrixFromFile(const std::string& filename) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<int>> matrix;
    int num;
    while (inputFile >> num) {
        matrix.push_back({num});
        while (inputFile.peek() == ' ') {
            inputFile.get();
            inputFile >> num;
            matrix.back().push_back(num);
        }
    }

    inputFile.close();
    return matrix;
}
void storeMatrixToFile(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (const auto& row : matrix) {
        for (int element : row) {
            outputFile << element << " ";
        }
        outputFile << std::endl;
    }

    outputFile.close();
}
vector<vector<int>>kernel={
    {1,1,1}, 
    {1,1,1},
    {1,1,1}
};




vector<vector<int>> Convolution(vector<vector<int>> img2){
    long r=img2.size(),c=img2[0].size();
    // int arr[r-2][c-2];
    

    int img[r][c];
    int arr[r][c];
    std::fill(&arr[0][0], &arr[0][0] + r * c, 0.0f);

    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            img[i][j]=img2[i][j];
            
        }
    }
    
    //////////////////////////////////inner convoultion , regardless of cloumn size : multipleof 16 or not , does well 

    /*

    for (int row = 1; row < r - 1; ++row) {
        for (int col = 1; col <= c - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
            __m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    // printf("%d %d\n",row,col);
                    // printf("--- %d %d\n",(row + ky) , (col + kx));
                    __m512 pixels = _mm512_loadu_ps(&img[row + ky][col + kx]);
                    
                    __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                }
            }
            _mm512_storeu_ps(&arr[row][col], sum);


        }
    }//remaing part <16 
    if((c-2)%16!=0){
       
        int rem=(c-2)%16;
        int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 
        for (int row = 1; row < r - 1; ++row) {
            //  __m512 pixels = _mm512_loadu_ps(&img[row][remain_single_part_start_col]);
            __m512 sum = _mm512_setzero_ps(); 


            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    // printf("%d %d\n",row,remain_single_part_start_col);
                    // printf("--- %d %d\n",(row + ky) , (remain_single_part_start_col + kx));
                    __m512 pixels = _mm512_loadu_ps(&img[row + ky][remain_single_part_start_col + kx]);
                    
                    __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                }
            }
            // _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
            int elements[16];
            _mm512_storeu_ps(elements, sum);   
            for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
                arr[row][remain_single_part_start_col+j]=elements[j];
                // printf("%d!\n",elements[j]);
            }

            

            
        }

    }
    */
    // #pragma omp single
    {
    int block_size=32;
    for (int row_sb = 1; row_sb <= r-2 ; row_sb += block_size) {
        for (int col_sb = 1; col_sb <= c-2; col_sb += block_size) {

            printf("++  %d  %d \n",row_sb,col_sb);
            int tot_rows = std::min(block_size, (int)r - row_sb);
            int tot_cols = std::min(block_size, (int)c - col_sb);

            for (int row = row_sb; row < row_sb+tot_rows; ++row) {
                printf("row %d \n",row);
                for (int col = col_sb; col <= col_sb+tot_cols+1 -17 ; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds

                    printf("  col  %d\n",col);
                    __m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            // printf("%d %d\n",row,col);
                            // printf("--- %d %d\n",(row + ky) , (col + kx));
                            __m512 pixels = _mm512_loadu_ps(&img[(row + ky)][col + kx]);
                            
                            __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                            sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                        }
                    }
                    // _mm512_storeu_ps(&arr[row][col], sum);
                    _mm512_storeu_ps(&arr[row][col], sum);


                }
            }



        }
    }


    if((c-2)%16!=0){
       
        int rem=(c-2)%16;
        int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 
        for (int row = 1; row < r - 1; ++row) {
            //  __m512 pixels = _mm512_loadu_ps(&img[row][remain_single_part_start_col]);
            __m512 sum = _mm512_setzero_ps(); 


            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    // printf("%d %d\n",row,remain_single_part_start_col);
                    // printf("--- %d %d\n",(row + ky) , (remain_single_part_start_col + kx));
                    __m512 pixels = _mm512_loadu_ps(&img[row + ky][remain_single_part_start_col + kx]);
                    
                    __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                }
            }
            // _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
            int elements[16];
            _mm512_storeu_ps(elements, sum);   
            for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
                arr[row][remain_single_part_start_col+j]=elements[j];
                // printf("%d!\n",elements[j]);
            }

            

            
        }

    }
    }


    //////////////////////////////////inner convoultion , regardless of cloumn size : multipleof 16 or not , does well above



    /////////////// only corners 
    arr[0][0]=  img[0][0]*kernel[1][1] + img[0][1]*kernel[1][2] + img[1][0]*kernel[2][1] + img[1][1]*kernel[2][2] ;

    arr[0][c-1]=  img[0][c-2]*kernel[1][0] + img[0][c-1]*kernel[1][1] + img[1][c-2]*kernel[2][0] + img[1][c-1]*kernel[2][1] ;


    arr[r-1][0]=  img[r-2][0]*kernel[0][1] + img[r-2][1]*kernel[0][2] + img[r-1][0]*kernel[1][1] + img[r-1][1]*kernel[1][2] ;

    
    arr[r-1][c-1]=  img[r-2][c-2]*kernel[0][0] + img[r-2][c-1]*kernel[0][1] + img[r-1][c-2]*kernel[1][0] + img[r-1][c-1]*kernel[1][1] ;

    //////////////


    /////row=0 excluding corner
    {
        int row_0=0;
        for (int col = 1; col <= c - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
                __m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
                for (int ky = 0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
                    for (int kx = -1; kx <= 1; ++kx) {
                        // printf("%d %d\n",row_0,col);
                        // printf("--- %d %d\n",(row_0 + ky) , (col + kx));
                        __m512 pixels = _mm512_loadu_ps(&img[row_0 + ky][col + kx]);
                        
                        __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                        sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                    }
                }
                _mm512_storeu_ps(&arr[row_0][col], sum);


        }
        if((c-2)%16!=0){
            int rem=(c-2)%16;
            int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 
                __m512 sum = _mm512_setzero_ps(); 

            for (int ky =0; ky <= 1; ++ky) { //statrt ky=0 not ky=-1
                for (int kx = -1; kx <= 1; ++kx) {
                    
                    __m512 pixels = _mm512_loadu_ps(&img[row_0 + ky][remain_single_part_start_col + kx]);
                    
                    __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                }
            }
            // _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
            int elements[16];
            _mm512_storeu_ps(elements, sum);   
            for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
                arr[row_0][remain_single_part_start_col+j]=elements[j];
                // printf("%d!\n",elements[j]);
            }
        }


    }
    /////row=r-1  excluding corner
    {
        int row_r_1=r-1;
        for (int col = 1; col <= c - 17; col += 16) { // Process 16 pixels per iteration, ensuring we stay within bounds
                __m512 sum = _mm512_setzero_ps();          //Works correct takes 16 from start each iter, iff possible
                for (int ky = -1; ky <= 0; ++ky) {  //ky =1 not possible
                    for (int kx = -1; kx <= 1; ++kx) {
                        // printf("%d %d\n",row_r_1,col);
                        // printf("--- %d %d\n",(row_r_1 + ky) , (col + kx));
                        __m512 pixels = _mm512_loadu_ps(&img[row_r_1 + ky][col + kx]);
                        
                        __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                        sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                    }
                }
                _mm512_storeu_ps(&arr[row_r_1][col], sum);


        }
        if((c-2)%16!=0){
            int rem=(c-2)%16;
            int remain_single_part_start_col=((int)(c-2)/16)*16+1;//1+ for dest matrix posn 
                __m512 sum = _mm512_setzero_ps(); 

            for (int ky =-1; ky <= 0; ++ky) { //ky =1 not possible
                for (int kx = -1; kx <= 1; ++kx) {
                    
                    __m512 pixels = _mm512_loadu_ps(&img[row_r_1 + ky][remain_single_part_start_col + kx]);
                    
                    __m512 filterVal = _mm512_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm512_fmadd_ps(pixels, filterVal, sum);
                }
            }
            // _mm512_storeu_ps(&arr[row][remain_single_part_start_col], sum); do this for only elements in row
            int elements[16];
            _mm512_storeu_ps(elements, sum);   
            for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
                arr[row_r_1][remain_single_part_start_col+j]=elements[j];
                // printf("%d!\n",elements[j]);
            }
        }


    }
    


    auto start2 = std::chrono::high_resolution_clock::now();
    



    // for(int i=0;i<2;i++){
    //     for(int j=0;j<r;j++){
    //         cout<<cop[i][j]<<" ";
    //     }cout<<"\n";
    // }

    
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<r;j++){
    //         cout<<cop2[i][j]<<" ";
    //     }cout<<"\n";
    // }

    

    

    {   /////// for col=0 ,  , row: 1->r-1


        int col_0=0;
        // for(int i=1;i<=r-2;i++){
        //     arr[i][col_0]=  cop[col_0][i-1]*kernel[0][1] + cop[col_0+1][i-1]*kernel[0][2] + cop[col_0][i]*kernel[1][1] + cop[col_0+1][i]*kernel[1][2]  +  cop[col_0][i+1]*kernel[2][1]  + cop[col_0+1][i+1]*kernel[2][2] ;
        // }

        float cop[2][r];  //truing to reuce cache miss, by transposing 
        for(int i=0;i<2;i++){
            for(int j=0;j<r;j++){
                cop[i][j]=img[j][i];
            }
        }

        int row_0=0; //its col_0
        float *temp=new float[r];
        for(int k=0;k<r;k++){temp[k]=0;}
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
        #pragma omp single nowait
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
                for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
                    // arr[row_0][remain_single_part_start_col+j]=elements[j];
                    // data[row_0 *r +remain_single_part_start_col+j]=elements[j];
                    temp[remain_single_part_start_col+j] =elements[j];
                    // printf("%d!\n",elements[j]);
                }
            }
        }

        for(int jk=1;jk<=r-2;jk++){
            arr[jk][0]=temp[jk];
        }







    }

    {   
        //col = c-1
        float cop2[2][r];  //truing to reuce cache miss, by transposing 
                    
        for(int j=0;j<r;j++){
            cop2[0][j]=img[j][c-1];
            cop2[1][j]=img[j][c-2];
        }

        int col_c_1=0;
        // for(int i=1;i<=r-2;i++){
        //     arr[i][c-1]=  cop2[col_c_1+1][i-1]*kernel[0][0] + cop2[col_c_1][i-1]*kernel[0][1] + cop2[col_c_1+1][i]*kernel[1][0] + cop2[col_c_1][i]*kernel[1][1]  +  cop2[col_c_1+1][i+1]*kernel[2][0]  + cop2[col_c_1][i+1]*kernel[2][1] ;
        // }  //Explanded with SIMD below***



        //
        int row_0=0; //its col_0
        float *temp=new float[r];
        for(int k=0;k<r;k++){temp[k]=0;}
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
        #pragma omp single nowait
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
                for(int j=0;j<rem;j++){ //to avoid junk values,also made independent for pragma by not updating other's part not update
                    // arr[row_0][remain_single_part_start_col+j]=elements[j];
                    // data[row_0 *r +remain_single_part_start_col+j]=elements[j];
                    temp[remain_single_part_start_col+j] =elements[j];
                    // printf("%d!\n",elements[j]);
                }
            }
        }






        for(int jk=1;jk<=r-2;jk++){
            arr[jk][c-1]=temp[jk];
        }        

        
    }










    std::vector<std::vector<int>> convo(r, std::vector<int>(c));
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            convo[i][j]=arr[i][j];
        }
    }
    return convo;
}




int main(){
     std::string filenam = "in.txt";
    
    /////////////////
    vector<vector<int>> img=readMatrixFromFile(filenam);
    long r=img.size(),c=img[0].size();
    std::cout<<"c.cpp "<<"r "<<r<<"  c "<<c<<endl;

    // disp(img);

    vector<vector<int>>convo= Convolution(img);
    std::string filename1 = "convo.txt";
    storeMatrixToFile(convo, filename1);


    std::string filename2 = "brute.txt";
    std::vector<std::vector<int>> brute(r, std::vector<int>(c,0));


    for(std::int32_t k = 0; k < r * c; k++) {
        float sum = 0.0;

        int i = k / c, j = k % c;
        for(int di = -1; di <= 1; di++)
            for(int dj = -1; dj <= 1; dj++) {
                
                int ni = i + di, nj = j + dj;
                if(ni >= 0 and ni < r and nj >= 0 and nj < c){ 
                    sum += kernel[di+1][dj+1] * img[ni][nj];
                    // if(i==0 and j==0){
                    //     printf("%d %d   %d-\n",di,dj,sum);
                    // }
                }
            }
        // sol_fs.write(reinterpret_cast<char*>(&sum), sizeof(sum));
        brute[i][j]=sum;
    }

    storeMatrixToFile(brute, filename2);

    

    

}