import numpy as np

def gf2_inverse(matrix):
    """Calculates the inverse of a matrix over GF(2).

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        The inverse matrix, or None if the matrix is singular.
    """

    n = len(matrix)
    augmented_matrix = np.hstack((matrix, np.eye(n))).astype(int)
    #print(f"augmented_matrix=\n{augmented_matrix}")
    try:
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

            #print(f"augmented_matrix=\n{augmented_matrix}")
            # Eliminate entries below the pivot
            for j in range(i + 1, n):
                if augmented_matrix[j, i] == 1:
                    augmented_matrix[j] ^= augmented_matrix[i]
                    #print(f"hello0")
            #print(f"hello 0.1")
    except Exception:
        print(Exception)
    #print(f"hello1")
    #print(f"augmented_matrix=\n{augmented_matrix}")
    
    # Back substitution
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if augmented_matrix[j, i] == 1:
                augmented_matrix[j] ^= augmented_matrix[i]
    #print("hello2")

    # Extract the inverse matrix
    inverse_matrix = augmented_matrix[:, n:]
    #print("hello3")
    #print(f"inverse_matrix=\n{inverse_matrix}")
    return inverse_matrix

# Example usage:
matrix = np.array(
[[1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
 [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
 [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
 [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
 [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
 [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
 [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
 [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]], dtype=int)

inverse_matrix = gf2_inverse(matrix)
#print(inverse_matrix)