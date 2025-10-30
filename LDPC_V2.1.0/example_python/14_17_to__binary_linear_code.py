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
#######################################################
import ldpc.codes, ldpc.code_util
H = ldpc.codes.hamming_code(3)
G = ldpc.code_util.construct_generator_matrix(H)
print(f"\nParity check matrix=\n{H.toarray()}")
print(f"\nGenerator matrix=\n{G.toarray()}")
#######################################################
print()
temp = H@G.T
print(f"temp = H@G.T=\n{temp.toarray()}")
temp.data = temp.data % 2
print()
print(f"temp.data % 2=\n{temp.toarray()}")
#######################################################
import numpy as np
b = np.array([1,0,1,1])
print(f"\nmessage={b}")
c = G.T@b % 2
print(f"\nc={c}")
#######################################################
syndrome=H@c % 2
print(f"\nsyndrome={syndrome}")