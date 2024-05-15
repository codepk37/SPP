import numpy as np

# Function to read matrix from a text file
def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix = [[float(num) for num in line.split()] for line in lines]
    return np.array(matrix)

# Function to compare two matrices
def compare_matrices(matrix1, matrix2):
    return np.array_equal(matrix1, matrix2)

# Read matrices A and B from files
A = read_matrix_from_file('1.txt')
B = read_matrix_from_file('2.txt')

# Perform matrix multiplication
result = np.dot(A, B)

# Read result matrix from file
expected_result = read_matrix_from_file('out.txt')

# Compare result with expected result
if compare_matrices(result, expected_result):
    print("Correct.")
else:
    print("Incorrect  ")
