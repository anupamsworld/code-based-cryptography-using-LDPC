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

ldpc_codeUtil_spec = importlib.util.spec_from_file_location("ldpc.code_util.code_util", CONFIG_ldpc_dir+"/src_python/ldpc/code_util/code_util.py")
ldpc_codeUtil = importlib.util.module_from_spec(ldpc_codeUtil_spec)
sys.modules["ldpc.code_util.code_util"] = ldpc_codeUtil
ldpc_codeUtil_spec.loader.exec_module(ldpc_codeUtil)


ldpc_codeUtilLegacyV1_spec = importlib.util.spec_from_file_location("ldpc.code_util._legacy_v1", CONFIG_ldpc_dir+"/src_python/ldpc/code_util/_legacy_v1.py")
ldpc_codeUtilLegacyV1 = importlib.util.module_from_spec(ldpc_codeUtilLegacyV1_spec)
sys.modules["ldpc.code_util._legacy_v1"] = ldpc_codeUtilLegacyV1
ldpc_codeUtilLegacyV1_spec.loader.exec_module(ldpc_codeUtilLegacyV1)


from ldpc.code_util.code_util import *
from ldpc.code_util._legacy_v1 import *