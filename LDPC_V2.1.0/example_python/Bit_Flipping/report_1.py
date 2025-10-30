from Hard_Message_Passing import *

# Example usage:
n = 20  # Code length
k = 5   # Number of information bits
m = n-k
d_c = 3   # Variable node degree
d_r = 4   # Check node degree


#H = generate_gallager_ldpc_code2(n, k, d_c, d_r)
H=np.array(
[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
 [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]]
)
print(f"type of H={type(H)}")
e=1
#error_bit_permutations=itertools.permutations(range(n), r=e)
error_bit_permutations=itertools.combinations(range(n), r=e)
#print(f"len(list(error_bit_permutations))={len(list(error_bit_permutations))}")
#print(f"error_bit_permutations=\n{list(error_bit_permutations)}")

import csv
from ldpc import BpDecoder
bpd = BpDecoder(
    H, #the parity check matrix
    error_rate=0.1, # the error rate on each bit
    max_iter=10000, #the maximum iteration depth for BP
    bp_method="product_sum", #BP method. The other option is `minimum_sum'
)
import os
PWD=os.path.dirname(os.path.realpath(__file__))
fileName = PWD+'/report_1/20-3-4_'+str(e)+'bit-error.csv'

headers = ['sent-message', 'received-message', 'HMP-decoded-value', 'HMP-syndrome', 'HMP-category', 'BP-decoded-value', 'BP-syndrome', 'BP-category']
count=0
with open(fileName, mode='w', newline='') as csvFile:
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(headers)
    for error_bit_pattern in error_bit_permutations:
        temp_message = [0] * n # this is equivalent to received vector
        #print(f"Type of temp_message={type(temp_message)}\ntemp_message={temp_message}")
        for error_bit_index in error_bit_pattern:
            temp_message[error_bit_index] = 1
        codeword_bits, check_bits = formTannerGraph(H)
        noOfIteration_MAX = 10000
        noOfIteration = 1
        decodeWithHMP(codeword_bits, check_bits, temp_message, noOfIteration, noOfIteration_MAX)
        HMP_decoded_value = [codeword_bits[codewordNode]['bit'] for codewordNode in codeword_bits]
        HMP_syndrome = (HMP_decoded_value@H.T) % 2
        #print(f"shape of temp_message={np.array([temp_message]).shape}")
        BP_decoded_value = bpd.decode(np.array(temp_message))
        BP_syndrome = (BP_decoded_value@H.T) % 2
        csvRow = [''.join(str(bit) for bit in [0]*n), ''.join(str(bit) for bit in temp_message), ''.join(str(bit) for bit in HMP_decoded_value), ''.join(str(bit) for bit in HMP_syndrome), getDecodedCategory(HMP_decoded_value, HMP_syndrome), ''.join(str(bit) for bit in BP_decoded_value), ''.join(str(bit) for bit in BP_syndrome), getDecodedCategory(BP_decoded_value, BP_syndrome)]
        csvWriter.writerow(csvRow)
        count += 1

print(f"count={count}")