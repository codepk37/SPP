import random
import sys
def generate_matrix(n, m):
    matrix = [[__*m+_ for _ in range(m)] for __ in range(n)]
    return matrix

def write_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

def main():
    # n = int(input("Enter the number of rows: "))
    # m = int(input("Enter the number of columns: "))
    if len(sys.argv) != 3:
        print("Usage: python script.py <number_of_rows> <number_of_columns>")
        sys.exit(1)

    n = int(sys.argv[1])
    m = int(sys.argv[2])

    matrix = generate_matrix(n, m)
    write_matrix_to_file(matrix, "in.txt")

    print(f"Matrix of size {n}x{m} generated and stored in 'in.txt'.")

if __name__ == "__main__":
    main()