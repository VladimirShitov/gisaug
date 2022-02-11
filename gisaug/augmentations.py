from typing import Sized, Union

import numpy as np
from scipy.interpolate import interp1d

from .validations import is_valid_coefficient, is_valid_probability, are_valid_bounds, are_valid_probability_bounds


class Augmentation:
    def _validate_parameters(self):
        raise NotImplementedError

    def __call__(self, x: np.array) -> np.array:
        raise NotImplementedError

    def visualize(self, x: np.array, vertical=True, figsize=(15, 5)):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=3, figsize=figsize)

        axes[0].plot(x)

        augmented_curve = self(x)
        axes[1].plot(augmented_curve)

        axes[2].plot(x, label="Original curve")
        axes[2].plot(augmented_curve, label="Augmented curve")
        axes[2].legend()

        return axes


class DropRandomPoints(Augmentation):
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

        self._validate_parameters()

    def _validate_parameters(self):
        try:
            is_valid_probability(self.p)
            self.should_generate_parameters = False
            self.keep_probability = self.p

        except (ValueError, TypeError):
            is_valid = are_valid_probability_bounds(self.p)
            self.should_generate_parameters = True

            if not is_valid:
                raise ValueError(f"{self.p} can't be validated as a parameter and is probably incorrect")

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

        if self.should_generate_parameters:
            self.keep_probability = np.random.uniform(self.p[0], self.p[1])

        keep_probabilities = np.random.uniform(size=len(x))

        return x[keep_probabilities < self.keep_probability]


class Stretch(Augmentation):
    """
    Augmentation, that makes the curve longer in the time direction by inserting new points. Value of
    points are interpolated from the closest observations

    Parameters
    ----------
    c : float or an iterable with 2 floats
        A stretching coefficient for the curve. E.g. if `c` is 3.6, the augmented curve will be 3.6 times longer
        than the original
    """

    @staticmethod
    def stretch_curve(x: np.array, stretching_coef: int) -> np.array:
        """Stretch `x` so that it's size will be `len(x) * stretching_coef`"""

        scope = np.arange(len(x))
        f = interp1d(scope, x)

        new_points = np.linspace(0, scope[-1], int(len(scope) * stretching_coef))
        interpolated_array = f(new_points)

        return interpolated_array

    def __init__(self, c: Union[float, Sized]):
        self.c = c

        self.should_generate_parameters = None
        self.stretching_coef = None

        self._validate_parameters()

    def _validate_parameters(self):
        try:
            is_valid_coefficient(self.c)
            self.should_generate_parameters = False
            self.stretching_coef = self.c

        except (ValueError, TypeError) as error:
            is_valid = are_valid_bounds(self.c)
            self.should_generate_parameters = True

            if not is_valid:
                raise ValueError(f"{self.c} can't be validated as a parameter and is probably incorrect")

    def __call__(self, x: np.array) -> np.array:
        """Make `x` longer by inserting interpolated points between the existing observations

        Parameters
        ----------
        x : array_like
            A numeric array with the curve to augment

        Returns
        -------
        numpy.array
            Augmented array with length ~ len(x) * c
        """

        if self.should_generate_parameters:
            self.stretching_coef = np.random.uniform(self.c[0], self.c[1])

        stretch_times = int(self.stretching_coef)
        augmented_array = self.stretch_curve(x, stretch_times)

        return augmented_array
