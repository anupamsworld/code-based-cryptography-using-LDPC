import importlib.util
import sys, os

PWD=os.path.dirname(os.path.realpath(__file__))
'''
ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../../../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)
'''
from ldpc.config import CONFIG_ldpc_dir

ldpc_codes_repCode_spec = importlib.util.spec_from_file_location("ldpc.codes.rep_code", CONFIG_ldpc_dir+"/src_python/ldpc/codes/rep_code.py")
ldpc_codes_repCode = importlib.util.module_from_spec(ldpc_codes_repCode_spec)
sys.modules["ldpc.codes.rep_code"] = ldpc_codes_repCode
ldpc_codes_repCode_spec.loader.exec_module(ldpc_codes_repCode)

ldpc_codes_hammingCode_spec = importlib.util.spec_from_file_location("ldpc.codes.hamming_code", CONFIG_ldpc_dir+"/src_python/ldpc/codes/hamming_code.py")
ldpc_codes_hammingCode = importlib.util.module_from_spec(ldpc_codes_hammingCode_spec)
sys.modules["ldpc.codes.hamming_code"] = ldpc_codes_hammingCode
ldpc_codes_hammingCode_spec.loader.exec_module(ldpc_codes_hammingCode)

ldpc_codes_randomBinaryCode_spec = importlib.util.spec_from_file_location("ldpc.codes.random_binary_code", CONFIG_ldpc_dir+"/src_python/ldpc/codes/random_binary_code.py")
ldpc_codes_randomBinaryCode = importlib.util.module_from_spec(ldpc_codes_randomBinaryCode_spec)
sys.modules["ldpc.codes.random_binary_code"] = ldpc_codes_randomBinaryCode
ldpc_codes_randomBinaryCode_spec.loader.exec_module(ldpc_codes_randomBinaryCode)

#sys.modules["ldpc.codes"]

from ldpc.codes.rep_code import rep_code, ring_code
from ldpc.codes.hamming_code import hamming_code
from ldpc.codes.random_binary_code import random_binary_code

print("upto here is fine")
#print("ldpc.codes.rep_code.__file__= "+str(rep_code.__file__))
print(rep_code(5))
