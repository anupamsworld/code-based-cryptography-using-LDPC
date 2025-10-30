#writing to file for string and matrix are placed in two different functions
#GPU support is introtuced

#cutomised for McEliece cryptography with LDPC code which was taken as reference from NASA to study same message trouble/attack.

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

import copy

import report_lib as rl

start_time = time.time()

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=100
noOfIteration_MAX = 10000



H = np.array(PCM.H_list)
G = np.array(GM.G_list)

print(f"Rank of H= {ldpc.mod2.rank(H)}")
print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")
print(f"ldpc.code_util.estimate_code_distance(H)={ldpc.code_util.estimate_code_distance(H)}")
#print(f"ldpc.code_util.compute_exact_code_distance(H)={ldpc.code_util.compute_exact_code_distance(H)}")
print(f"len(utl.find_all_zero_rows(H))={len(utl.find_all_zero_rows(H))}")
index, min_sum = utl.min_row_sum(H)
print(f"Row {index} has the minimum sum of {min_sum}")
print(f"sum(GPUC.GF2MatrixMultUsingGpu(H[0], H))= {sum(GPUC.GF2MatrixMultUsingGpu(H[0], H.T))}")
exit()
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

error_list = list()
messages_list = list()
messageG_star_list = list()
y_list = list()
pad_length = 10

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

duplicateErrorCheckVector = cupy.array([1]*n)
errorIndexesSet = set(range(n))
noMoreErrorPossible = False

for i_th_message in range(3):
    print(f"------i_th_message={i_th_message}------")
    print(f"i_th_message={i_th_message}")

    raw_message_list = [0]*(7154-pad_length)
    raw_message_list[0] = 1
    pad_list = [0]*(pad_length)
    #pad_list[0] = 1
    message_list = utl.padTheMessage(k, raw_message_list, pad_list)
    messages_list.append(message_list)
    message = np.array(message_list)

    raw_message = np.array(raw_message_list)
    pad = np.array(pad_list)

    #write_to_file(message, "message", 1)

    #codeword = (message@G)%2
    codeword = GPUC.GF2MatrixMultUsingGpu(message, G)


    
    
    '''
    fileDir = PWD+'/report_3/3_all_zero_codeword_and_random_error_bit/'
    #fileDir = PWD+'/report_1/3_all_zero_codeword_and_specific_codewords/'
    rl.decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report_2(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir)
    #rl.decode_All_Possible_Errors_With_HMP_and_BP_and_Generate_Report(H, n, k, m, noOfErrorBits, noOfIteration_MAX, fileDir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")
    exit()
    '''

    if str(utl.getMessageFromSystematicCodeword(n, k, codeword)) == str(message):
        print(f"Message retrival is successful.")
    else:
        print(f"Message retrival is unsuccessful.")

    #write_to_file(codeword, "codeword", 1)

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
    messageG_star_list.append(list(copy.deepcopy(messageG_star)))
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
    
    #print(f"errorIndexesSet={errorIndexesSet}")
    #print(f"type(errorIndexesSet)={type(errorIndexesSet)}")
    #print(f"range(n)={range(n)}")
    #print(f"type(range(n))={type(range(n))}")
    #exit()
    while True:
        e, errorIndexesSet, error_positions = GPUC.GF2makeErrorMatrixUsingGPU(n, bitErrorProb=error_rate, errorIndexesSet=errorIndexesSet, noOfErrorBits=82, method=3)
        
        weightWithoutError = cupy.sum(duplicateErrorCheckVector)
        weightError = cupy.sum(e)
        duplicateErrorCheckVector_temp = cupy.bitwise_xor(cupy.copy(duplicateErrorCheckVector), cupy.array(e))
        weightWithError = cupy.sum(duplicateErrorCheckVector_temp)

        print(f"\nweightWithoutError={weightWithoutError}\nweightError={weightError}\nweightWithError={weightWithError}")
        print(f"weightWithoutError-weightError={weightWithoutError-weightError}")

        if(weightWithoutError-weightError == weightWithError):
            duplicateErrorCheckVector = duplicateErrorCheckVector_temp
            break
        if len(errorIndexesSet) < 82:
            print(f"No more error possible to generate.")
            noMoreErrorPossible = True
            break
    print(f"e is made.")
    #print(f"type(e)= {type(e)}")
    error_list.append(list(e))
    #sum_e = sum(e)
    sum_e = cupy.sum(e)
    print(f"Number of 1's in the error vector(sum_e)= {sum_e}")
    #print(f"type(error_positions)= {type(error_positions)}")
    #print(f"error_positions= {error_positions}")
    error_rate = sum_e/n
    print(f"After forming error matrix with randomized bits, the error rate becomes= {error_rate}")
    #print(f"The error type(e)={type(e)}\ttype(sum_e)={type(sum_e)}\ttype(error_rate)={type(error_rate)}")

    #y = (messageG_star + e)%2
    y = GPUC.GF2MatrixAddUsingGPU(messageG_star, e)
    y_list.append(list(copy.deepcopy(y)))

    #print(f"type(y)={type(y)}\t type(P_inverse)={type(P_inverse)}")
    '''
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
    '''
    if noMoreErrorPossible:
        break

full_file_path = PWD+'/report_5_SameMessageTrouble/3_no_common_error_index/test-16_3messages.csv'
rl.sameMessageTroubleReport(n, k, pad_length, messages_list, messageG_star_list, error_list, full_file_path)

confirmedMessageValues = utl.findConfirmedMessageValues(n, k, y_list)
#print(f"confirmedMessagePositions={confirmedMessagePositions}")
#print(f"type(confirmedMessagePositions)={type(confirmedMessagePositions)}")
totalConfirmedMessagePositions = len(confirmedMessageValues)
print(f"Total confirmed message positions = {totalConfirmedMessagePositions}")

import csv
with open(full_file_path, mode='a', newline="") as csvFile:

    
    #csv.field_size_limit(sys.maxsize)
    #print(csv.field_size_limit())

    csvWriter = csv.writer(csvFile, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #confirmedMessagePositions_index = cupy.where(cupy.asarray(confirmedMessagePositions)==1)[0]
    confirmedMessageValues_indexAndValue = np.array(list(confirmedMessageValues.items()))
    csvRow = [str(f"Total confirmed message positions = {totalConfirmedMessagePositions}"), np.array2string(confirmedMessageValues_indexAndValue, threshold=np.inf)]
    csvWriter.writerow(csvRow)

#print(error_list)
#print(f"type(error_list)= {type(error_list)}")
#print(f"type(error_list[0])= {type(error_list[0])}")
#print(f"type(error_list[0][0])= {type(error_list[0][0])}")



end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time:.4f} seconds")