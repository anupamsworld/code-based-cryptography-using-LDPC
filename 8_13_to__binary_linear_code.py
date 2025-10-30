import importlib.util
import sys, os

import ldpc.mod2

PWD=os.path.dirname(os.path.realpath(__file__))

ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)

from ldpc.config import CONFIG_ldpc_dir

import logging.config
if os.path.isfile(CONFIG_ldpc_dir+'/logger.config'):
    logging.config.fileConfig(CONFIG_ldpc_dir+'/logger.config', defaults=None, disable_existing_loggers=True, encoding=None)
    _8Logger=logging.getLogger('tester')
    #codeUtilLogger.debug('logger.debug is working')
    #codeUtilLogger.info('logger.info is working')
    #codeUtilLogger.warning('logger.warning is working')
    #codeUtilLogger.critical('logger.critical is working')
    _8LoggerHandler=_8Logger.handlers
    _8Logger.info("_8Logger is initialized")
    print(_8LoggerHandler)

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
H = ldpc.codes.hamming_code(3)
print(H.toarray())
#######################################################
n = H.shape[1]
print(f"Number of physical bits, n = {n}")
#######################################################
k = ldpc.code_util.compute_code_dimension(H)
print(f"Number of logical bits, k = {k}")
#######################################################
rank=ldpc.mod2.rank(H)
print(f"rank={rank}")
#######################################################
d, number_code_words_sampled, lowest_weight_codewords = ldpc.code_util.estimate_code_distance(H, timeout_seconds = 0.1)
print(f"Code distance estimate, d <= {d} (no. codewords sampled: {number_code_words_sampled})")
#######################################################
n, k, d_estimate = ldpc.code_util.compute_code_parameters(H)
print(f"Code parameters: [n = {n}, k = {k}, d <= {d_estimate}]")
#######################################################
M = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]])

n, k, d_estimate = ldpc.code_util.compute_code_parameters(M)
print(f"Code parameters: [n = {n}, k = {k}, d <= {d_estimate}]")
#######################################################