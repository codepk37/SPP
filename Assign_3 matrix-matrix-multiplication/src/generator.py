import sys
import numpy as np

def generate_matrix(filename, rows, cols):
    matrix = np.random.uniform(1, 10, size=(rows, cols))  # Generate random matrix with elements in range [1, 3]
    np.savetxt(filename, matrix, fmt='%.d')  # Save matrix to file

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py n k m")
        return
    
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    m = int(sys.argv[3])

    generate_matrix('1.txt', n, k)
    generate_matrix('2.txt', k, m)

    print(f"Matrices of size {n}x{k} and {k}x{m} have been generated ")#and stored in '1.txt' and '2.txt' respectively.

if __name__ == "__main__":
    main()
