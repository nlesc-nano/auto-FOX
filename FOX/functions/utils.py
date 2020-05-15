import warnings

from ..utils import *
from .. import utils

__all__ = utils.__all__
__doc__ = utils.__doc__
del utils

_warning = DeprecationWarning("The 'FOX.functions.utils' module is deprecated; "
                              "use 'FOX.utils' from now on")
warnings.warn(_warning)
del _warning
