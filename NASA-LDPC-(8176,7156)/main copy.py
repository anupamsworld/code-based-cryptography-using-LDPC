#writing to file for string and matrix are placed in two different functions
#GPU support is introtuced

# LDPC package Version: 2.2.8

import parity_check_matrix as PCM
import generator_matrix as GM

import numpy as np
import os


import time

import sys
import cupy

PWD=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(PWD+'/../'))

import lib.GPU_Computations as GPUC
import lib.util as utl

import Bit_Flipping.Hard_Message_Passing as HMP
import lib.store_and_fetch as sf

import ldpc.mod2
import ldpc.code_util
np.set_printoptions(threshold=np.inf)
start_time = time.time()

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=50
noOfIteration_MAX = 10000

H = np.array(PCM.H_list)
G = np.array(GM.G_list)
exit()
fileDir = PWD+'/report_3/5_specific_codeword_and_random_error_bit/'
#fileDir = PWD+'/report_1/3_all_zero_codeword_and_specific_codewords/'

message, _, _ = GPUC.GF2makeErrorMatrixUsingGPU(7154, errorIndexesSet=set(range(7154)), noOfErrorBits=7000, method=3)
codeword = GPUC.GF2MatrixMultUsingGpu(message, G)

with open(fileDir + "message_3.txt", 'w') as codeword_report_file:
    codeword_report_file.write(f"{message}")
exit()
'''
with open(fileDir + "codeword.txt", 'w') as codeword_report_file:
    codeword_report_file.write(f"{codeword}")
exit()
'''


print(f"Rank of H= {ldpc.mod2.rank(H)}")
print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")
print(f"ldpc.code_util.estimate_code_distance(H)={ldpc.code_util.estimate_code_distance(H)}")
#exit()
'''
print(f"{find_all_zero_rows(H)}")
print(f"np.shape(H)= {np.shape(H)}")
print(f"{sum(H[511])}")
'''
#H_sparse = sf.matrix_to_sparse(H)
#H_sparse_str = sf.print_sparse_matrix(H)

#print(f"H_sparse_str = \n{H_sparse_str}")

#write_str_to_file(H_sparse_str, "H_sparse")

#exit()

#GxH_T = (G@H.T)%2

#write_to_file(GxH_T, "GxH_T", 2)

#message = np.array([0]*7154)
#message[0] = 1


#write_to_file(message, "message", 1)

#codeword = (message@G)%2



import report_lib as rl




message = sf.read_array_from_file(fileDir+'message.txt', element_type='int')
codeword = sf.read_array_from_file(fileDir+'codeword.txt', element_type='int')

'''
print("\ntype="+str(type(sf.read_array_from_file(fileDir+'message.txt', element_type='int'))))
print(f"message={sf.read_array_from_file(fileDir+'message.txt', element_type='int')}")
exit()
'''
rl.decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report_2(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir, originalCodeword=codeword)
#rl.decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")
exit()


if str(utl.getMessageFromSystematicCodeword(n, k, codeword)) == str(message):
    print(f"Message retrival is successful.")
else:
    print(f"Message retrival is unsuccessful.")

#write_to_file(codeword, "codeword", 1)






print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")





S = sf.fetch_matrix_from_file(PWD+"/S_matrix.txt")
#S = np.load(PWD+"/S_matrix.npy")

print(f"type(S)={type(S)}")
#print(f"rank(S)={ldpc.mod2.rank(S)}")

#print(f"np.linalg.det(S)={np.linalg.det(S)}\tBut we need mod2 determinant")

S_inverse = []
'''
if ldpc.mod2.rank(S) == k:
    print(f"S is invertible.")
    #S_inverse = np.linalg.inv(S).astype(int)
    S_inverse = gf2_inverse(S)
else:
    print(f"S is not invertible")
    exit()
'''
#S_inverse = gf2_inverse(S)
S_inverse = sf.fetch_matrix_from_file(PWD+"/S_inverse_matrix.txt")

P = sf.fetch_matrix_from_file(PWD+"/P_matrix.txt")
#P = np.load(PWD+"/P_matrix.npy")

'''
if ldpc.mod2.rank(P) == n:
    print(f"P is invertible.")
    #S_inverse = np.linalg.inv(S).astype(int)
    P_inverse = gf2_inverse(P)
else:
    print(f"P is not invertible")
    exit()
'''
#P_inverse = gf2_inverse(P)
P_inverse = sf.fetch_matrix_from_file(PWD+"/P_inverse_matrix.txt")

#messageS = (message@S)%2
messageS = GPUC.GF2MatrixMultUsingGpu(message, S)

#messageSG = (messageS@G)%2
messageSG = GPUC.GF2MatrixMultUsingGpu(messageS, G)

#messageG = (message@G)%2
messageG = GPUC.GF2MatrixMultUsingGpu(message, G)

#####################################
#G_star = (((S @ G) % 2) @ P) % 2

SG = GPUC.GF2MatrixMultUsingGpu(S, G)
G_star = GPUC.GF2MatrixMultUsingGpu(SG, P)
#####################################

#####################################
#messageG_star = (message@G_star) % 2

messageG_star = GPUC.GF2MatrixMultUsingGpu(message, G_star)
#####################################

error_rate=noOfErrorBits/n # the error rate on each bit
#error_rate=0.20,

#e = np.array([0]*8176)
'''
e[5] = 1
e[100] = 1
'''
'''
for index in range(k):
    if index % 9 == 0 :
        e[index] = 1
'''
'''
for index in range(k):
    None
'''

e, error_positions = GPUC.GF2makeErrorMatrixUsingGPU(n, bitErrorProb=error_rate)
print(f"e is made.")
#sum_e = sum(e)
sum_e = cupy.sum(e)
print(f"Number of 1's in the error vector(sum_e)= {sum_e}")
print(f"Number of 1's in the error vector(error_positions)= {error_positions}")
error_rate = sum_e/n
print(f"After forming error matrix with randomized bits, the error rate becomes= {error_rate}")
print(f"The error type(e)={type(e)}\ttype(sum_e)={type(sum_e)}\ttype(error_rate)={type(error_rate)}")

#y = (messageG_star + e)%2
y = GPUC.GF2MatrixAddUsingGPU(messageG_star, e)

print(f"type(y)={type(y)}\t type(P_inverse)={type(P_inverse)}")

#yp_inverse = (y@P_inverse) % 2
yp_inverse = GPUC.GF2MatrixMultUsingGpu(y, P_inverse)

from ldpc import BpDecoder
bpd = BpDecoder(
    #bpd = bp_decoder(
        H, #the parity check matrix
        error_rate=float(error_rate), # the error rate on each bit
        max_iter=noOfIteration_MAX, #the maximum iteration depth for BP
        bp_method="product_sum", #BP method. The other option is `minimum_sum'
        schedule = "serial" # the BP schedule
    )
messageSG_decoded = bpd.decode(np.array(yp_inverse)) # mS is the BP_decoded_codeword
print(f"messageSG_decoded category={HMP.getDecodedCategory(messageSG, messageSG_decoded, GPUC.GF2MatrixMultUsingGpu(messageSG_decoded, H.T))}")

decoded_messageS = None
#if sum((messageSG_decoded@H.T)%2) == 0:
if sum(GPUC.GF2MatrixMultUsingGpu(messageSG_decoded, H.T)) == 0:
    print(f"\n{messageSG_decoded} is a valid deoded codeword")
    decoded_messageS = utl.getMessageFromSystematicCodeword(n, k, messageSG_decoded)
    print(f"{len(decoded_messageS)}")
    print(f"Decrypted original messageS={decoded_messageS}")

    #decrypted_msg = ((np.array(list(decoded_messageS)).astype(int)) @ S_inverse) % 2
    decrypted_message = GPUC.GF2MatrixMultUsingGpu(np.array(list(decoded_messageS)).astype(int), S_inverse)
    print(f"The decryped message={decrypted_message}")


else:
    print(f"\n{messageSG_decoded} is not a valid deoded codeword")
    





end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time:.4f} seconds")