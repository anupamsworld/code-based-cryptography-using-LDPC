import importlib.util
import sys, os
PWD=os.path.dirname(os.path.realpath(__file__))

ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)

import ldpc.code_util
from ldpc.config import CONFIG_ldpc_dir


import numpy as np
import ldpc.codes
from ldpc import BpDecoder

H=ldpc.codes.rep_code(3) #parity check matrix for the length-3 repetition code
n=H.shape[1] #the codeword length

bpd = BpDecoder(
    H, #the parity check matrix
    error_rate=0.1, # the error rate on each bit
    max_iter=n, #the maximum iteration depth for BP
    bp_method="product_sum", #BP method. The other option is `minimum_sum'
)
G=ldpc.code_util.construct_generator_matrix(H)
n, k, d_estimate = ldpc.code_util.compute_code_parameters(H)
print(f"Code parameters: [n = {n}, k = {k}, d <= {d_estimate}]")
message=b = np.array([0])
#codeword=np.array([1,1,1])
codeword=(G.T@message) %2
received_vector=np.array([0,1,1])
decoded_codeword=bpd.decode(received_vector)
syndrome=(received_vector@H.T.toarray())%2
print(f"\ncodeword=\n{codeword}")
print(f"\nreceived_vector=\n{received_vector}")
print(f"\nParity Check Matrix=\n{H.toarray()}")
print(f"\nParity Check Matrix=\n{H}")
print(f"\nsyndrome=\n{syndrome}")
error=bpd.decode(syndrome)
print(f"\nError from Syndrome={error}")
print(f"\nDecoded codeword from syndrom based decoding={error+received_vector}")
print(f"\ndecoded_codeword=\n{decoded_codeword}")
print(f"\n(decoded_codeword@H.T)%2=\n{(decoded_codeword@H.T)%2}")