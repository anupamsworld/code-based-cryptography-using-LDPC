import importlib.util
import sys, os
PWD=os.path.dirname(os.path.realpath(__file__))

ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)

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

error = np.array([0,1,1])
syndrome = H@error %2 # the syndrome of the error

decoding=bpd.decode(syndrome)

print(f"Error: {error}")
print(f"Syndrome: {syndrome}")
print(f"Decoding: {decoding}")