import numpy as np
import copy
def generate_binary_strings(n):
    """Generates all 2^n binary strings of length n.

    Args:
    n: The length of the binary strings to generate.

    Returns:
    A list of all 2^n binary strings of length n.
    """

    if n == 0:
        return [""]

    binary_strings = []
    for string in generate_binary_strings(n - 1):
        binary_strings.append(string + "0")
        binary_strings.append(string + "1")

    return binary_strings

def makeMessageCodewordTable(k, G):
    all_msg = generate_binary_strings(k)
    #print(f"all_msg=\n{all_msg}")
    messageCodewordTable = {}
    for msg in all_msg:
        codeword = (np.array(list(copy.deepcopy(msg)), dtype=int) @ G) % 2
        messageCodewordTable[''.join(map(str, list(codeword)))] = copy.deepcopy(msg)
    return messageCodewordTable

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

def getMessageFromCodeword(msgSG_decoded, messageCodewordTable):
    return messageCodewordTable[''.join(map(str, list(msgSG_decoded)))]