
run:
	gcc -O0 mat-vec.c -o matp_0
	gcc -O1 mat-vec.c -o matp_1 
	gcc -O2 mat-vec.c -o matp_2
	gcc -O3 mat-vec.c -o matp_3 

	./matp_0>> report1.txt
	./matp_1>> report1.txt
	./matp_2>> report1.txt
	./matp_3>> report1.txt


runo:
	gcc -O0 mat-vec.c -o matp_1
	gcc -O1 mat-vec.c -o matp_2 
	gcc -O3 -march=native -ftree-vectorize -funroll-loops -ffast-math -mavx2 -mfma mat-vec.c -o matp_3
	gcc -O3 -fopenmp mat-vec.c -o matp_4
	gcc -O3 -fopenmp -march=native -ftree-vectorize -funroll-loops -ffast-math -mavx2 -mfma mat-vec.c -o matp_5

	./matp_1     > 1.txt
	./matp_2     > 2.txt
	./matp_3     > 3.txt
	./matp_4     > 4.txt
	./matp_5     > 5.txt 






clean:
	rm matp_1 matp_2 matp_3 matp_4 matp_5
