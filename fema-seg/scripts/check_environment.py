import json
import platform
import sys
import torch

result={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda}
if torch.cuda.is_available():result['gpu']=torch.cuda.get_device_name()
try:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    result['mamba_import']='ok'
except ImportError as e:result['mamba_import']=str(e)
print(json.dumps(result,indent=2))
