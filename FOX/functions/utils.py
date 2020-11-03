"""Placeholder."""

import warnings

from ..utils import *  # noqa: F401, F403
from .. import utils  # noqa: F401

__all__ = utils.__all__
__doc__ = utils.__doc__
del utils

_warning = FutureWarning("The 'FOX.functions.utils' module is deprecated; "
                         "use 'FOX.utils' from now on")
warnings.warn(_warning)
del _warning
