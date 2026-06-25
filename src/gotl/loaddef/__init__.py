"""LoadDef OTL parameter generation.

Wraps the LoadDef library (Martens et al.) to compute ocean tidal
loading coefficients from Earth models and tide models via MPI
subprocess calls.
"""

from gotl.loaddef.batch import OTLGenResult, generate_otl_params
from gotl.loaddef.env import LoadDefEnv, load_loaddef_env

__all__ = [
    "generate_otl_params",
    "OTLGenResult",
    "load_loaddef_env",
    "LoadDefEnv",
]