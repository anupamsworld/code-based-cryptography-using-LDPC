import importlib.util
import sys, os, time
PWD=os.path.dirname(os.path.realpath(__file__))

ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)

import ldpc.codes.hamming_code
from ldpc.config import CONFIG_ldpc_dir


import numpy as np
import ldpc.codes
import ldpc.mod2
import ldpc.code_util

def write_to_file(array_or_list, filename, dimention):
    PWD=os.path.dirname(os.path.realpath(__file__))

    if dimention==2:
        with open(PWD+'/'+filename+'.txt', 'w') as file:
            for i in range(array_or_list.shape[0]):
                for j in range(array_or_list.shape[1]):
                    file.write(str(array_or_list[i][j]))
                file.write("\n")
    elif dimention==1:
        with open(PWD+'/'+filename+'.txt', 'w') as file:
            for i in range(array_or_list.shape[0]):
                file.write(str(array_or_list[i]))

PWD=os.path.dirname(os.path.realpath(__file__))

'''

H_rep=ldpc.codes.rep_code(4) #parity check matrix for the length-3 repetition code
n=H_rep.shape[1] #the codeword length

G_rep=ldpc.code_util.construct_generator_matrix(H_rep.toarray())
n_rep, k_rep, d_estimate_rep = ldpc.code_util.compute_code_parameters(H_rep)
row_echelon_H_rep=ldpc.mod2.row_echelon(H_rep)[0]
reduced_row_echelon_H_rep=ldpc.mod2.reduced_row_echelon(H_rep)[0]



H_ham=ldpc.codes.hamming_code(4) #parity check matrix for the length-3 repetition code
n=H_ham.shape[1] #the codeword length

G_ham=ldpc.code_util.construct_generator_matrix(H_ham.toarray())
n_ham, k_ham, d_estimate_ham = ldpc.code_util.compute_code_parameters(H_ham)
row_echelon_H_ham=ldpc.mod2.row_echelon(H_ham)[0]
reduced_row_echelon_H_ham=ldpc.mod2.reduced_row_echelon(H_ham)[0]



print(f"Code parameters: [n_rep = {n_rep}, k_rep = {k_rep}, d_rep <= {d_estimate_rep}]")
print(f"\nH_rep=\n{H_rep.toarray()}")
print(f"\nrow_echelon_H_rep=\n{row_echelon_H_rep}")
print(f"\nreduced_row_echelon_H_rep=\n{reduced_row_echelon_H_rep}")
print(f"\nG_rep=\n{G_rep.toarray()}")

print(f"Code parameters: [n_ham = {n_ham}, k_ham = {k_ham}, d_ham <= {d_estimate_ham}]")
print(f"\nH_ham=\n{H_ham.toarray()}")
print(f"\nrow_echelon_H_ham=\n{row_echelon_H_ham}")
print(f"\nreduced_row_echelon_H_ham=\n{reduced_row_echelon_H_ham}")
print(f"\nG_ham=\n{G_ham.toarray()}")


##### Now lets encode, decode, correct error #####
'''

matrix_1 = np.array(
[
[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
]
)

print(f"ldpc.mod2.rank(matrix_1) = {ldpc.mod2.rank(matrix_1)}")


matrix_2 = np.array(
[
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
)

print(f"ldpc.mod2.rank(matrix_1) = {ldpc.mod2.rank(matrix_2)}")


myList = [[1,2,3],[4,5,6],[7,8,9]]

myList[1:2][0:2] = [112, 222]

#print(f"{myList}")






'''
def string_to_int_list(string):
  """Converts a string of integers separated by spaces into a list of integers.

  Args:
    string: The input string containing integers.

  Returns:
    A list of integers.
  """
  return list(map(int, string.split("")))

# Example usage:
integer_string = "12345"
integer_list = string_to_int_list(integer_string)
print(integer_list)  # Output: [1, 2, 3, 4, 5]
'''

def binary_string_to_int_list_no_spaces(string):
  """Converts a string of integers without spaces into a list of integers.

  Args:
    string: The input string containing integers without spaces.

  Returns:
    A list of integers.
  """
  result = []
  for char in string:
    if char.isdigit():
      result.append(int(char))
  return result

# Example usage:
integer_string = "12345"
integer_list = binary_string_to_int_list_no_spaces(integer_string)
#print(integer_list)  # Output: [12345]


def fetch_matrix_from_file(fullFilePath, fetchType="allAtOnce"):
    import numpy as np

    list = []
    if fetchType == "rowAtOnce":
      with open(fullFilePath, "r") as file:
            for line in file:
                # Process each line here
                #print(line.strip())  # Remove leading/trailing whitespace
                list.append(binary_string_to_int_list_no_spaces(line.strip()))
    elif fetchType == "allAtOnce":
      with open(fullFilePath, "r") as file:
        file_content = file.read()
      
      # Split the string into a list of lines
      lines = file_content.splitlines()

      # Iterate over each line
      for line in lines:
          # Process each line here
          list.append(binary_string_to_int_list_no_spaces(line.strip()))

    matrix = np.array(list)

    print(f"Formed matrix has shape={matrix.shape}")

    return matrix


def generate_random_permutation_matrix(n):
  """Generates a random permutation matrix of size NxN.

  Args:
    n: The size of the permutation matrix.

  Returns:
    A NumPy array representing the random permutation matrix.
  """

  permutation = np.random.permutation(n)
  matrix = np.zeros((n, n), dtype=int)
  for i, j in enumerate(permutation):
    matrix[i, j] = 1
  return matrix

def gf2_inverse(matrix):
    """Calculates the inverse of a matrix over GF(2).

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        The inverse matrix, or None if the matrix is singular.
    """

    n = len(matrix)
    augmented_matrix = np.hstack((matrix, np.eye(n))).astype(int)
    for i in range(n):
        # Find the pivot row
        pivot_row = i
        while pivot_row < n and augmented_matrix[pivot_row, i] == 0:
            pivot_row += 1

        if pivot_row == n:
            return None  # Matrix is singular

        # Swap rows if necessary
        if pivot_row != i:
            augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            if augmented_matrix[j, i] == 1:
                augmented_matrix[j] ^= augmented_matrix[i]
    
    # Back substitution
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if augmented_matrix[j, i] == 1:
                augmented_matrix[j] ^= augmented_matrix[i]

    # Extract the inverse matrix
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix

import numpy as np
from scipy.sparse import csr_matrix

def matrix_to_sparse(matrix):
    """
    Converts a dense matrix to a sparse matrix in CSR (Compressed Sparse Row) format.

    Args:
      matrix: A 2D NumPy array representing the dense matrix.

    Returns:
      A SciPy CSR sparse matrix.
    """

    rows, cols = matrix.shape
    data = []
    row_indices = []
    col_indices = []

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:
                data.append(matrix[i, j])
                row_indices.append(i)
                col_indices.append(j)

    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
    return sparse_matrix

#S = np.random.randint(2, size=(7154,7154))

#write_to_file(S, "S_matrix", 2)

#np.save(PWD+"/S_matrix.npy", S)
import cupy
start_time = time.time()
S1 = np.random.randint(2, size=(7154,7154))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to generate S1 matrix: {elapsed_time:.4f} seconds")
'''
start_time = time.time()
s2 = cupy.random.randint(2, size=(7154,7154))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to generate S2 matrix using cupy: {elapsed_time:.4f} seconds")
'''
write_to_file(S1, "S_matrix_3", 2)

exit()


#P = generate_random_permutation_matrix(8176)

#write_to_file(P, "P_matrix", 2)

#np.save(PWD+"/P_matrix.npy", P)



'''
S = fetch_matrix_from_file(PWD+"/S_matrix.txt")
#S = fetch_matrix_from_file(PWD+"/S_matrix.txt", "rowAtOnce")

start_time = time.time()
S_inverse = gf2_inverse(S)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to calculate S_inverse: {elapsed_time:.4f} seconds")

write_to_file(S_inverse, "S_inverse_matrix", 2)

P = fetch_matrix_from_file(PWD+"/P_matrix.txt")
#P = fetch_matrix_from_file(PWD+"/P_matrix.txt", "rowAtOnce")

start_time = time.time()
P_inverse = gf2_inverse(P)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to calculate P_inverse: {elapsed_time:.4f} seconds")

write_to_file(P_inverse, "P_inverse_matrix", 2)
'''



'''

import cupy
arr = cupy.array([1,1,0,1,0])
indexes = cupy.where(arr)
#print(f"indexes=\n{indexes}")
arr2 = cupy.array([1,1,0,1,0])
arr3 = cupy.asnumpy(cupy.bitwise_xor(cupy.array(arr), cupy.array(arr2)))

print(f"{indexes[0]}")
print(f"{str(indexes[0])}")

print(f"arr=\n{arr}")
print(f"arr2=\n{arr2}")
print(f"arr3=\n{arr3}")

print(f"type(arr3)= {type(arr3)}")

'''

rng = range(2,-1,-1)
print(f"{rng}")



# Example usage:
dense_matrix = np.array([[1, 0, 0],
                         [0, 2, 0],
                         [0, 0, 3],
                         [4, 0, 0]])

sparse_matrix = matrix_to_sparse(dense_matrix)
print(sparse_matrix) 


def print_sparse_matrix(matrix):
    rows, cols = matrix.shape
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            if matrix[i, j] != 0:
                row_str += f"({i}, {j}) {matrix[i, j]} "
        if row_str:
            print(row_str)

print_sparse_matrix(sparse_matrix)