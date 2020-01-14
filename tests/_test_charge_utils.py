"""A module for testing :mod:`FOX.functions.charge_utils` ."""

from pathlib import Path

import numpy as np
import pandas as pd
from assertionlib import assertion

from FOX.functions.charge_utils import assign_constraints

PATH = Path('tests') / 'test_files'


def test_assign_constraints() -> None:
    """Test :func:`assign_constraints`."""
    df = pd.DataFrame(index=['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'])
    df['constraints'] = None

    constraints = [
        'H < 1',
        'C > 2.0'
        '3.0 > N',
        '4 < O',
        '1 < F < 2.0',
        '2 > P > 1.0'
    ]
