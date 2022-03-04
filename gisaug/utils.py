import numpy as np


def numbers_array_to_string(numbers: np.array, round_digits: int = 2) -> str:
    """Convert array with numbers to a nicely looking string

    Examples
    --------
    >>> numbers_array_to_string(np.array([0.123, 0.857, 0.406]))
    "0.12, 0.86, 0.41"
    """
    return ", ".join(map(str, numbers.round(round_digits)))
