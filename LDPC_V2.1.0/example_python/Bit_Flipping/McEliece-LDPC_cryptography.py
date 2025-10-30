import numpy as np
import copy

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

# Example usage:
n = 20  # Code length
k = 10   # Number of information bits
m = n-k
#d_c = 3   # Variable node degree
#d_r = 4   # Check node degree

noOfErrorBits=1
noOfIteration_MAX = 10000

#H = generate_gallager_ldpc_code2(n, k, d_c, d_r)
H=np.array(
[
[1, 0, 0, 0, 0,  0, 0, 0, 0, 0,  1, 0, 1, 0, 0,  0, 1, 0, 1, 1],
[0, 1, 0, 0, 0,  0, 0, 0, 0, 0,  0, 1, 0, 1, 0,  1, 0, 1, 0, 1],
[0, 0, 1, 0, 0,  0, 0, 0, 0, 0,  1, 0, 0, 1, 0,  0, 1, 1, 0, 1],
[0, 0, 0, 1, 0,  0, 0, 0, 0, 0,  0, 0, 1, 0, 1,  1, 1, 0, 1, 0],
[0, 0, 0, 0, 1,  0, 0, 0, 0, 0,  0, 1, 0, 0, 1,  1, 0, 1, 1, 0],

[1, 0, 1, 0, 0,  0, 1, 0, 1, 1,  1, 0, 0, 0, 0,  0, 0, 0, 0, 0],
[0, 1, 0, 1, 0,  1, 0, 1, 0, 1,  0, 1, 0, 0, 0,  0, 0, 0, 0, 0],
[1, 0, 0, 1, 0,  0, 1, 1, 0, 1,  0, 0, 1, 0, 0,  0, 0, 0, 0, 0],
[0, 0, 1, 0, 1,  1, 1, 0, 1, 0,  0, 0, 0, 1, 0,  0, 0, 0, 0, 0],
[0, 1, 0, 0, 1,  1, 0, 1, 1, 0,  0, 0, 0, 0, 1,  0, 0, 0, 0, 0]
]
)
#print(f"type of H={type(H)}")
import ldpc.code_util
print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")


G = np.array(
[[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],       
 [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],       
 [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],       
 [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],       
 [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],       
 [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],       
 [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],       
 [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],       
 [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],       
 [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
)

print(f"G=\n{G}")
print(f"G@H.T=\n{(G@H.T)%2}")

S = np.array(
[[1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
 [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
 [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
 [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
 [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
 [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
 [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
 [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]]
)
# this is invertible

S_inverse = []
print(f"rank(S)={ldpc.mod2.rank(S)}")
print(f"np.linalg.det(S)={np.linalg.det(S)}\tBut we need mod2 determinant")
if ldpc.mod2.rank(S) == k:
    print(f"S is invertible.")
    #S_inverse = np.linalg.inv(S).astype(int)
    S_inverse = gf2_inverse(S)
else:
    print(f"S is not invertible")

print(f"S_inverse=\n{S_inverse}")
print(f"(S@S_inverse)%2=\n{(S@S_inverse)%2}")

P = np.array(
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)
print(f"P=\n{P}")

P_inverse = gf2_inverse(P)
print(f"p_inverse=\n{P_inverse}")
print(f"(P@P_inverse)%2=\n{(P@P_inverse)%2}")

#msg = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0])
#msg = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1]) # given by sir
msg = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1]) # given by sir


print(f"msg=\n{msg}")

msgS = (msg@S)%2
print(f"msgS=\n{msgS}")

msgSG = (msgS@G)%2
print(f"msgSG=\n{msgSG}")

msgG = (msg@G)%2
print(f"msgG=\n{msgG}")


G_star = (((S @ G) % 2) @ P) % 2
print(f"G_star=\n{G_star}")

msgG_star = (msg@G_star) % 2
print(f"msgG_star=\n{msgG_star}")

e = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(f"e=\n{e}")

y = (msgG_star + e)%2
print(f"y=\n{y}")

yp_inverse = (y@P_inverse) % 2
print(f"yp_inverse=\n{yp_inverse}")

from ldpc import BpDecoder
bpd = BpDecoder(
    #bpd = bp_decoder(
        H, #the parity check matrix
        error_rate=noOfErrorBits/n, # the error rate on each bit
        max_iter=noOfIteration_MAX #the maximum iteration depth for BP
        #bp_method="product_sum", #BP method. The other option is `minimum_sum'
    )

msgSG_decoded = bpd.decode(np.array(yp_inverse)) # mS is the BP_decoded_codeword
print(f"\nmsgSG_decoded=\n{msgSG_decoded}")
print(f"type(msgSG_decoded)={type(msgSG_decoded)}\t {np.shape(msgSG_decoded)}")

from codingCryptoBasic import *

decoded_msgS = None
messageCodewordTable = makeMessageCodewordTable(k, G)
if sum((msgSG_decoded@H.T)%2) == 0:
    print(f"\n{msgSG_decoded} is a valid deoded codeword")
    decoded_msgS = getMessageFromCodeword(msgSG_decoded, messageCodewordTable)
    print(f"Decrypted original msgS={decoded_msgS}")

print(f"type(np.array(list(decoded_msgS)))={type(np.array(list(decoded_msgS)))}")
print(f"{np.shape(np.array(list(decoded_msgS)))}")
decrypted_msg = ((np.array(list(decoded_msgS)).astype(int)) @ S_inverse) % 2
print(f"The decryped message={decrypted_msg}")

print("\n\n\n")

msg5 = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1]) #


print(f"msg5=\n{msg5}")

msg5S = (msg5@S)%2
print(f"msg5S=\n{msg5S}")