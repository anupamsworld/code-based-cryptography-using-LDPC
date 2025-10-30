import parity_check_matrix as PCM
import generator_matrix as GM

import numpy as np
import os
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

def getMessageFromSystematicCodeword(n, k, codeword):
        return codeword[n-k,n]

import time
start_time = time.time()


H = np.array(PCM.H_list)
G = np.array(GM.G_list)

#GxH_T = (G@H.T)%2

#write_to_file(GxH_T, "GxH_T", 2)

message = np.array([0]*7154)
message[0] = 1

#write_to_file(message, "message", 1)

codeword = (message@G)%2

#write_to_file(codeword, "codeword", 1)

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=1
noOfIteration_MAX = 10000

import ldpc.code_util
print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")



import lib.store_and_fetch as sfe
PWD=os.path.dirname(os.path.realpath(__file__))


S = sfe.fetch_matrix_from_file(PWD+"/S_matrix.txt")

print(f"type(S)={type(S)}")
print(f"rank(S)={ldpc.mod2.rank(S)}")

print(f"np.linalg.det(S)={np.linalg.det(S)}\tBut we need mod2 determinant")

S_inverse = []

if ldpc.mod2.rank(S) == k:
    print(f"S is invertible.")
    #S_inverse = np.linalg.inv(S).astype(int)
    S_inverse = gf2_inverse(S)
else:
    print(f"S is not invertible")


P = sfe.fetch_matrix_from_file(PWD+"/P_matrix.txt")

P_inverse = gf2_inverse(P)

messageS = (message@S)%2

messageSG = (messageS@G)%2

messageG = (message@G)%2

#SG = (S @ G) % 2
G_star = (((S @ G) % 2) @ P) % 2
#G_star = (SG @ P) % 2

messageG_star = (message@G_star) % 2

e = np.array([0]*8176)
e[5] = 1

y = (messageG_star + e)%2
'''
yp_inverse = (y@P_inverse) % 2

from ldpc import BpDecoder
bpd = BpDecoder(
    #bpd = bp_decoder(
        H, #the parity check matrix
        error_rate=noOfErrorBits/n, # the error rate on each bit
        max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
        bp_method="product_sum", #BP method. The other option is `minimum_sum'
    )
messageSG_decoded = bpd.decode(np.array(yp_inverse)) # mS is the BP_decoded_codeword

decoded_messageS = None
if sum((messageSG_decoded@H.T)%2) == 0:
    print(f"\n{messageSG_decoded} is a valid deoded codeword")
    decoded_messageS = getMessageFromSystematicCodeword(n, k, messageSG_decoded)
    print(f"{len(decoded_messageS)}")
    print(f"Decrypted original messageS={decoded_messageS}")
else:
    print(f"\n{messageSG_decoded} is not a valid deoded codeword")
'''

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time:.4f} seconds") 