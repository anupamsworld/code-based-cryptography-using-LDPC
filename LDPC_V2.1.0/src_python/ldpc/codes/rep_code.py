import scipy.sparse as sp
import numpy as np


import sys, os
import importlib.util
#print("full path--> "+os.path.dirname(os.path.realpath(__file__)))
'''
PWD=os.path.dirname(os.path.realpath(__file__))
ldpc_config_spec = importlib.util.spec_from_file_location("ldpc.config", PWD+"/../../../config.py")
ldpc_config = importlib.util.module_from_spec(ldpc_config_spec)
sys.modules["ldpc.config"] = ldpc_config
ldpc_config_spec.loader.exec_module(ldpc_config)
'''
from ldpc.config import CONFIG_ldpc_dir
import logging.config
if os.path.isfile(CONFIG_ldpc_dir+'/logger.config'):
    logging.config.fileConfig(CONFIG_ldpc_dir+'/logger.config', defaults=None, disable_existing_loggers=True, encoding=None)
    repCodeLogger=logging.getLogger('repCode')
    #codeUtilLogger.debug('logger.debug is working')
    #codeUtilLogger.info('logger.info is working')
    #codeUtilLogger.warning('logger.warning is working')
    #codeUtilLogger.critical('logger.critical is working')
    repCodeLoggerHandler=repCodeLogger.handlers
    print(repCodeLoggerHandler)


def rep_code(distance: int) -> sp.csr_matrix:
    """
    Outputs repetition code parity check matrix for specified distance.
    Parameters
    ----------
    distance: int
        The distance of the repetition code. Must be greater than or equal to 2.
    Returns
    -------
    sp.csr_matrix
        The repetition code parity check matrix in sparse CSR matrix format.
    Examples
    --------
    >>> print(rep_code(5).toarray())
    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]]
    """
    repCodeLogger.info("inside repcode function")
    repCodeLogger.debug("inside repcode function")
    if distance < 2:
        raise ValueError("Distance should be greater than or equal to 2.")

    rows = []
    cols = []
    data = []

    for i in range(distance - 1):
        rows += [i, i]
        cols += [i, i+1]
        data += [1, 1]

    return sp.csr_matrix((data, (rows, cols)), shape=(distance-1, distance), dtype=np.uint8)

def ring_code(distance: int) -> sp.csr_matrix:
    """
    Outputs ring code (closed-loop repetion code) parity check matrix
    for a specified distance. 
    Parameters
    ----------
    distance: int
        The distance of the repetition code. Must be greater than or equal to 2.
    Returns
    -------
    sp.csr_matrix
        The repetition code parity check matrix in sparse CSR matrix format.
    Examples
    --------
    >>> print(ring_code(5).toarray())
    [[1 1 0 0 0]
     [0 1 1 0 0]
     [0 0 1 1 0]
     [0 0 0 1 1]
     [1 0 0 0 1]]
    """

    if distance < 2:
        raise ValueError("Distance should be greater than or equal to 2.")

    rows = []
    cols = []
    data = []

    for i in range(distance - 1):
        rows += [i, i]
        cols += [i, i+1]
        data += [1, 1]

    # close the loop
    rows += [distance - 1, distance - 1]
    cols += [0, distance - 1]
    data += [1, 1]

    return sp.csr_matrix((data, (rows, cols)), shape=(distance, distance), dtype=np.uint8)


