from collections import Iterable
from numbers import Number
from typing import Sized


class WrongProbabilityValueError(ValueError):
    message = "{} is not a valid probability value. Please, provide a number between 0 and 1"

    def __init__(self, value):
        error_message = self.message.format(value)
        super().__init__(error_message)


def is_valid_probability(p: Number) -> bool:
    """Check if `p` is a valid value for probability"""

    if not isinstance(p, Number):
        raise TypeError(f"{type(p)} is not a valid type for probability! Please, provide a number between 0 and 1")

    if not 0 <= p <= 1:
        raise WrongProbabilityValueError

    return True


def are_valid_bounds(bounds: Sized) -> bool:
    """Check if `bounds` can be valid interval bounds"""

    error_message = f"{bounds} are not valid bounds"

    if not isinstance(bounds, Iterable):
        raise TypeError(f"{error_message}. Please, provide an iterable (e.g. tuple or a list) with 2 elements")

    if len(bounds) != 2:
        raise ValueError(f"{error_message}. Please, provide exactly 2 numbers")

    if not isinstance(bounds[0], Number) or not isinstance(bounds[1], Number):
        raise TypeError(f"{error_message}. Bounds must be numeric")

    if bounds[1] < bounds[0]:
        raise ValueError(f"{error_message}. Lower bound must be less or equal to the upper bound")

    return True


def are_valid_probability_bounds(probs: Sized) -> bool:
    return are_valid_bounds(probs) and is_valid_probability(probs[0]) and is_valid_probability(probs[1])
