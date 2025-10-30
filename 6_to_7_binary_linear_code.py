import importlib.util
import sys, os

PWD=os.path.dirname(os.path.realpath(__file__))

ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)

from ldpc.config import CONFIG_ldpc_dir



'''
ldpc_code_util_spec = importlib.util.spec_from_file_location("ldpc.code_util", CONFIG_ldpc_dir+"/src_python/ldpc/code_util/__init__custom.py")
ldpc_code_util = importlib.util.module_from_spec(ldpc_code_util_spec)
sys.modules["ldpc.code_util"] = ldpc_code_util
ldpc_code_util_spec.loader.exec_module(ldpc_code_util)
'''
'''
ldpc_codes_spec = importlib.util.spec_from_file_location("ldpc.codes", CONFIG_ldpc_dir+"/src_python/ldpc/codes/__init__custom.py")
ldpc_codes = importlib.util.module_from_spec(ldpc_codes_spec)
sys.modules["ldpc.codes"] = ldpc_codes
ldpc_codes_spec.loader.exec_module(ldpc_codes)
'''

 

import numpy as np
import ldpc.codes
import ldpc.code_util
n=5 #specifies the lenght of the repetition code
H=ldpc.codes.rep_code(n) #returns the repetition code parity check matrix
#from ldpc.codes.rep_code import rep_code
#from ldpc.codes import *
#from ldpc.code_util import *
#import ldpc.codes
#H=rep_code(n)
print(H)
print(H.toarray())