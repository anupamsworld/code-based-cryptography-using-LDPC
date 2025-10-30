def gf2_inverse(matrix):
    """Calculates the inverse of a matrix over GF(2).

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        The inverse matrix, or None if the matrix is singular.
    """

    import numpy as np

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

def matrix_to_sparse(matrix):
    """
    Converts a dense matrix to a sparse matrix in CSR (Compressed Sparse Row) format.

    Args:
      matrix: A 2D NumPy array representing the dense matrix.

    Returns:
      A SciPy CSR sparse matrix.
    """

    from scipy.sparse import csr_matrix

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

def print_sparse_matrix(matrix):
    rows, cols = matrix.shape
    row_str = ""
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:
                row_str += f"({i}, {j}) {matrix[i, j]}\n"
    return row_str

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

def find_all_zero_rows(matrix):
    """
    Finds the row numbers where all elements are zero in a given matrix.

    Args:
    matrix: A NumPy array representing the matrix.

    Returns:
    A list of row numbers where all elements are zero.
    """
    import numpy as np
    all_zero_rows = np.where(np.all(matrix == 0, axis=1))[0]
    return all_zero_rows.tolist()

def min_row_sum(matrix):
    '''
    if not matrix:
        return None  # Return None if the matrix is empty
    '''
    if matrix.size == 0:  # Check if the matrix is empty
        return None 
    min_sum = float('inf')
    min_row_index = -1
    
    for i, row in enumerate(matrix):
        row_sum = sum(row)
        if row_sum < min_sum:
            min_sum = row_sum
            min_row_index = i
    
    return min_row_index, min_sum

def getMessageFromSystematicCodeword(n, k, codeword):
        return codeword[:k,]
def padTheMessage(k:int, raw_message:list, pad:list)->list:
    if len(raw_message) + len(pad) == k:
        return raw_message+pad
    else:
        return []

def findConfirmedMessageValues(n, k, receivedMessages:list, full_file_path=None):
    import itertools
    import cupy
    messageCombinations = itertools.combinations(range(len(receivedMessages)), r=2)
    #confirmedMessagePositions_cupyArray = cupy.array([1]*n)
    confirmedMessageValues = {} # dictionary is used to store the confirmed message values and so that we can check if there is any conflict in the confirmed message values or whether the confirmed message values are present or not.
    #print(f"confirmedMessagePositions.shape= {confirmedMessagePositions.shape}")
    for messageCombinationIndex, messageCombination in enumerate(messageCombinations):

        if(len(receivedMessages[messageCombination[0]]) == n and len(receivedMessages[messageCombination[1]]) == n):
            messagePositions = cupy.logical_not(cupy.bitwise_xor(cupy.array(receivedMessages[messageCombination[0]]), cupy.array(receivedMessages[messageCombination[1]])))
            for positionIndex, messageBit in enumerate(messagePositions):
                if messageBit == 1:
                    if positionIndex not in confirmedMessageValues:
                        confirmedMessageValues[positionIndex] = receivedMessages[messageCombination[0]][positionIndex]
                    elif confirmedMessageValues[positionIndex] != receivedMessages[messageCombination[0]][positionIndex]:
                        raise Exception(f"Error: confirmedMessageValues[positionIndex]= {confirmedMessageValues[positionIndex]}, messageBit= {messageBit}")
            
            
            #after_xor = cupy.bitwise_xor(cupy.array(receivedMessages[messageCombination[0]]), cupy.array(receivedMessages[messageCombination[1]]))
            #print(f"after_xor= {after_xor}")
            #after_not = cupy.logical_not(after_xor)
            #print(f"after_not= {after_not}")
            #after_and = cupy.bitwise_and(after_not, cupy.array(confirmedMessagePositions))
            #print(f"after_and= {after_and}")
            #confirmedMessagePositions = list(cupy.asnumpy(after_and))
            #print(f"type(confirmedMessagePositions)= {type(confirmedMessagePositions)}")
            #print(f"confirmedMessagePositions= {confirmedMessagePositions}")

        else:
            print(f"Error: len(receivedMessages[messageCombination[0]])= {len(receivedMessages[messageCombination[0]])}, len(receivedMessages[messageCombination[1]])= {len(receivedMessages[messageCombination[1]])}, n= {n}")
            #confirmedMessagePositions = []
        '''
        if full_file_path is not None:
            with open(full_file_path, "a") as file:
                file.write(f"messageCombinationIndex= {messageCombinationIndex},  Total confirmed message positions = {len(confirmedMessageValues)}\n")
        '''
    '''
    confirmedMessageValues_list = [0]*n
    for positionIndex, messageBit in confirmedMessageValues.items():
        confirmedMessageValues_list[positionIndex] = messageBit
    #confirmedMessagePositions_cupyArray = cupy.asarray(confirmedMessageValues_list)

    return confirmedMessageValues_list
    '''
    return confirmedMessageValues

def text_to_binary(text: str, bits: int = 8) -> str:
    """
    Convert a string into its binary representation.

    :param text: The input string.
    :param bits: Number of bits per character (default 8 for standard ASCII).
    :return: A space-separated string of binary numbers.
    """
    return ''.join(format(ord(char), f'0{bits}b') for char in text)

def binary_to_text(binary_str: str) -> str:
    """
    Convert a continuous binary string (e.g., '0100100001100101') to text.
    Assumes 8 bits per character.
    """
    # Split into 8-bit chunks
    chars = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
    # Convert each chunk to a character
    return ''.join(chr(int(b, 2)) for b in chars)

def hex_to_binary(hex_str: str) -> str:
    """
    Convert a hexadecimal string to its binary representation.
    Each hex digit becomes 4 binary bits.
    """
    # Remove optional '0x' prefix and convert to lowercase
    hex_str = hex_str.strip().lower().replace("0x", "")
    return ''.join(format(int(c, 16), '04b') for c in hex_str)

def binary_to_hex(binary_str: str) -> str:
    """
    Convert a continuous binary string (e.g., '0100100001100101') to hexadecimal.
    Assumes binary string length is a multiple of 4.
    """
    # Pad binary string to multiple of 4 bits (if needed)
    padded = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
    # Split into 4-bit chunks and convert
    hex_str = ''.join(format(int(padded[i:i+4], 2), 'x') for i in range(0, len(padded), 4))
    return hex_str