import numpy as np

def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split()))
            matrix.append(row)
    return np.array(matrix)  # Convert to NumPy array

def are_matrices_equal(matrix1, matrix2):
    return np.array_equal(matrix1, matrix2)

def main():
    # Read matrices from files
    input_matrix = read_matrix_from_file("convo.txt")
    # print("Input Matrix:")
    

    output_matrix = read_matrix_from_file("brute.txt")
    # print("Output Matrix:")
    # print(output_matrix)

    # Check if transpose is correct
    if are_matrices_equal(input_matrix, output_matrix):
        print("convolution is correct.")
    else:
        print("convolution is incorrect.")

if __name__ == "__main__":
    main()