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

def binary_matrix_to_hex_numpy(matrix):
    hex_rows = []
    for row in matrix:
        # Convert row to binary string (e.g., [1 0 1 1] → '1011')
        binary_str = ''.join(row.astype(str))
        # Convert to int, then to hex
        hex_val = hex(int(binary_str, 2))[2:].zfill((len(binary_str) + 3) // 4)
        hex_rows.append(hex_val.upper())
    return hex_rows

start_time = time.time()

# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=50
noOfIteration_MAX = 10000


import report_lib as rl

fileDir = PWD+'/report_3/5_specific_codeword_and_random_error_bit/'
#fileDir = PWD+'/report_1/3_all_zero_codeword_and_specific_codewords/'
np.set_printoptions(threshold=np.inf)

message1 = sf.read_array_from_file(fileDir+'message.txt', element_type='int')
codeword = sf.read_array_from_file(fileDir+'codeword.txt', element_type='int')


S1 = sf.fetch_matrix_from_file(PWD+"/S_matrix.txt")
S2 = sf.fetch_matrix_from_file(PWD+"/S_matrix_2.txt")
S3 = sf.fetch_matrix_from_file(PWD+"/S_matrix_3.txt")
#print(f"binary_matrix_to_hex_numpy(S1)={binary_matrix_to_hex_numpy(S1)}")

msgS1 = (message1@S1)%2
msgS2 = (message1@S2)%2
msgS3 = (message1@S3)%2

# find hamming disdance between msgS1 and msgS2
hamming_distance_message1_msgS1 = np.sum(message1 != msgS1)
hamming_distance_message1_msgS2 = np.sum(message1 != msgS2)
hamming_distance_message1_msgS3 = np.sum(message1 != msgS3)
print(f"Distance between Message_1 and Scambled Message, scrabled with S1: {hamming_distance_message1_msgS1}")
print(f"Distance between Message_1 and Scambled Message, scrabled with S2: {hamming_distance_message1_msgS2}")
print(f"Distance between Message_1 and Scambled Message, scrabled with S3: {hamming_distance_message1_msgS3}")
print(f"Distance between Scrambled Message_1 and Scrambled Message, scrabled with S1: {np.sum(msgS1 != msgS2)}")
print(f"Distance between Scrambled Message_1 and Scrambled Message, scrabled with S2: {np.sum(msgS1 != msgS3)}")
print(f"Distance between Scrambled Message_1 and Scrambled Message, scrabled with S3: {np.sum(msgS2 != msgS3)}")


# Now we will see how different message1s are scrambled by same S matrix
message2 = sf.read_array_from_file(fileDir+'message_2.txt', element_type='int')
message3 = sf.read_array_from_file(fileDir+'message_3.txt', element_type='int')
msg2S1 = (message2@S1)%2
msg3S1 = (message3@S1)%2
hamming_distance_message2_msgS1 = np.sum(message2 != msg2S1)
hamming_distance_message3_msgS1 = np.sum(message3 != msg3S1)
print(f"Distance between Message_2 and Scambled Message, scrabled with S1: {hamming_distance_message2_msgS1}")
print(f"Distance between Message_3 and Scambled Message, scrabled with S1: {hamming_distance_message3_msgS1}")

