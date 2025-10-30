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


# Example usage:
n = 8176  # Code length
k = 7154   # Number of information bits
m = n-k

noOfErrorBits=50
noOfIteration_MAX = 10000

H = np.array(PCM.H_list)
#G = np.array(GM.G_list)

print(f"shape of H: {H.shape}")

# Count the number of 1's -1 in each row of H and sum them up withand witout GPU
def count_xor_operations_cpu(matrix):
    xor_count_cpu = np.sum(matrix, axis=1) - 1
    return np.sum(xor_count_cpu)
def count_xor_operations_gpu(matrix):
    matrix_gpu = cupy.asarray(matrix)
    xor_count_gpu = cupy.sum(matrix_gpu, axis=1) - 1
    return cupy.sum(xor_count_gpu)

start_time = time.time()
print("Counting XOR operations using GPU:")
print("Ones count using GPU:", count_xor_operations_gpu(H))
print(f"Time taken using GPU: {time.time() - start_time:.4f} seconds")

start_time = time.time()
print("Counting XOR operations using CPU:")
print("Ones count using CPU:", count_xor_operations_cpu(H))
print(f"Time taken using CPU: {time.time() - start_time:.4f} seconds")
 