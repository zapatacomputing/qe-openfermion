from ._io import *
from ._utils import *
import warnings

warnings.warn(
    """qe-openfermion will be fully deprecated \
starting 2/1/2021. As of 12/15/2020, all functionality and future development \
related to the qe-openfermion will be contained within zquantum.core.openfermion.""",
)
