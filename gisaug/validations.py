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


def are_valid_probability_bounds(probs: Sized) -> bool:
    error_message = f"{probs} are not valid probability bounds"

    if not isinstance(probs, Iterable):
        raise TypeError(f"{error_message}. Probabilities must be iterable (e.g. tuple or a list) with 2 elements")

    if len(probs) != 2:
        raise ValueError(f"{error_message}. Please, provide exactly 2 numbers for probabilities bounds")

    if probs[1] < probs[0]:
        raise ValueError(f"{error_message}. Probability lower bound must be less or equal to the upper bound")

    return is_valid_probability(probs[0]) and is_valid_probability(probs[1])
