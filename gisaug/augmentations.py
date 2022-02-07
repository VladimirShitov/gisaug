from numbers import Number
from typing import Sized, Union

import numpy as np

from validations import is_valid_probability, are_valid_probability_bounds


class DropRandomPoints:
    """
    Augmentation, that randomly drops points from time series to make it shorter in the time, but generally
    preserve shape

    Parameters
    ----------
    p : float or tuple of 2 floats
        A percentage of points to keep. A single number represents the proportion of the augmented curve length
        to the original curve length. If tuple of two numbers is provided (e.g. (`a`, `b`)), the proportion will
        be chosen randomly and uniformly between `a` and `b` on each run.
    """
    def __init__(self, p: Union[float, tuple]):
        self.p = p

        self.should_generate_parameters = None
        self.keep_probability = None

    def _validate_parameters(self):
        if is_valid_probability(self.p):
            self.should_generate_parameters = False
            self.keep_probability = self.p

        elif are_valid_probability_bounds(self.p):
            self.should_generate_parameters = True

        else:
            raise ValueError(f"{self.p} can't be validated and probably contains wrong values.")

    def __call__(self, x: np.array) -> np.array:
        """Make `x` shorter by dropping random points

        Parameters
        ----------
        x : array_like
            A sequence of elements to augment. Can contain elements of any type, as this augmentation is type agnostic

        Returns
        -------
        numpy.array
            Augmented array with length ~ p * len(x)
        """

        if self.should_generate_parameters():
            self.keep_probability = np.random.uniform(self.p[0], self.p[1])

        keep_probabilities = np.random.uniform(size=len(x))

        return x[keep_probabilities < self.keep_probability]
