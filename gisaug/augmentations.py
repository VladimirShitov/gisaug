from typing import Sized, Union

import numpy as np
from scipy.interpolate import interp1d

from .validations import is_valid_coefficient, is_valid_probability, are_valid_bounds, are_valid_probability_bounds, \
                         is_valid_positive_integer
from .utils import numbers_array_to_string


class Augmentation:
    def _validate_parameters(self):
        raise NotImplementedError

    def __call__(self, x: np.array) -> np.array:
        raise NotImplementedError

    def visualize(self, x: np.array, vertical=True, figwidth=8, figheight=8):
        import matplotlib.pyplot as plt

        augmented_curve = self(x)

        if vertical:
            fig, axes = plt.subplots(ncols=3, figsize=(figheight, figwidth))

            original_curve_time = np.arange(len(x))
            axes[0].plot(x, original_curve_time)
            axes[0].invert_xaxis()
            axes[0].invert_yaxis()

            augmented_curve_time = np.arange(len(augmented_curve))
            axes[1].plot(augmented_curve, augmented_curve_time)
            axes[1].invert_xaxis()
            axes[1].invert_yaxis()

            axes[2].plot(x, original_curve_time, label="Original curve")
            axes[2].plot(augmented_curve, augmented_curve_time, label="Augmented curve")
            axes[2].invert_xaxis()
            axes[2].invert_yaxis()

        else:
            fig, axes = plt.subplots(nrows=3, figsize=(figwidth, figheight))
            axes[0].plot(x)
            axes[1].plot(augmented_curve)

            axes[2].plot(x, label="Original curve")
            axes[2].plot(augmented_curve, label="Augmented curve")

        axes[0].set_title("Original curve")
        axes[1].set_title("Augmented curve")

        axes[2].legend(loc="upper right")
        axes[2].set_title("Curves together")

        return fig, axes


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

    def visualize(self, x: np.array, vertical=True, figwidth=8, figheight=8):
        fig, axes = super().visualize(x, vertical, figwidth, figheight)
        fig.suptitle(f"p: {round(self.keep_probability, 3)}", fontweight="bold")
        fig.tight_layout()

        return axes


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

        stretch_times = self.stretching_coef
        augmented_array = self.stretch_curve(x, stretch_times)

        return augmented_array

    def visualize(self, x: np.array, vertical=True, figwidth=8, figheight=8):
        fig, axes = super().visualize(x, vertical, figwidth, figheight)
        fig.suptitle(f"c: {round(self.stretching_coef, 3)}", fontweight="bold")
        fig.tight_layout()

        return axes


class DropRandomRegions(Augmentation):
    """
    Augmentation, that randomly drops regions from time series to make it shorter in the time direction

    Parameters
    ----------
    p : float or tuple of 2 floats
        A proportion of the length of the dropping region to the original curve length. E.g. p=0.5 means that half of
        the curve will be dropped. `p` can be a single number or iterable with 2 elements.
        If tuple of two numbers is provided (e.g. (`a`, `b`)), the proportion will be chosen randomly and uniformly
        between `a` and `b` on each run.
    k : int or tuple with 2 ints
        Number of regions to drop. E.g. if p=0.5 and k=2 than half of the curve will be dropped first, and than another
        half of the new curve will be dropped. Generally, augmented curve length is proportional to p^k of the
        original curve length.
    """
    @staticmethod
    def remove_region(x: np.array, region_start: int, region_end: int) -> np.array:
        """Remove region from `region_start` to `region_end` from `x`"""
        new_array = x[:region_start].tolist()
        new_array.extend(x[region_end:].tolist())

        return np.array(new_array)

    @staticmethod
    def remove_random_region(x: np.array, region_length: int) -> np.array:
        """Remove random region of length `region_size` from `x`"""

        dropping_start = int(np.random.uniform(0, len(x) - region_length))

        return DropRandomRegions.remove_region(
            x, region_start=dropping_start, region_end=dropping_start + region_length
        )

    def __init__(self, p, k):
        self.p = p
        self.k = k

        self.should_generate_p = None
        self.should_generate_k = None
        self.dropping_proportions = None
        self.regions_dropped = None

        self._validate_parameters()

    def _validate_parameters(self):
        try:
            is_valid_positive_integer(self.k)
            self.should_generate_k = False
            self.regions_dropped = self.k

        except (ValueError, TypeError):
            is_valid = are_valid_bounds(self.k)
            self.should_generate_k = True

            if not is_valid:
                raise ValueError(f"{self.k} can't be validated as a parameter k and is probably incorrect")

        try:
            is_valid_probability(self.p)
            self.should_generate_p = False

        except (ValueError, TypeError):
            is_valid = are_valid_probability_bounds(self.p)
            self.should_generate_p = True

            if not is_valid:
                raise ValueError(f"{self.p} can't be validated as a parameter p and is probably incorrect")

    def __call__(self, x: np.array) -> np.array:
        """Make `x` shorter by removing random regions from it

        Parameters
        ----------
        x : array_like
            A numeric array with the curve to augment

        Returns
        -------
        numpy.array
            Augmented array with length ~ len(x) * p^k
        """

        if self.should_generate_k:
            self.regions_dropped = np.random.randint(self.k[0], self.k[1])

        if self.should_generate_p:
            self.dropping_proportions = np.random.uniform(low=self.p[0], high=self.p[1], size=self.regions_dropped)
        else:
            self.dropping_proportions = np.ones(shape=self.regions_dropped) * self.p

        augmented_array = self.remove_random_region(x, region_length=int(len(x) * self.dropping_proportions[0]))

        for i in range(1, self.regions_dropped):
            dropping_region_length = int(len(augmented_array) * self.dropping_proportions[i])

            augmented_array = self.remove_random_region(
                augmented_array, region_length=dropping_region_length
            )

        return augmented_array

    def visualize(self, x: np.array, vertical=True, figwidth=8, figheight=8):
        fig, axes = super().visualize(x, vertical, figwidth, figheight)

        # Convert array of probabilities to string: e.g [0.123, 0.857, 0.406] -> "0.12, 0.86, 0.41"
        dropping_probabilities = numbers_array_to_string(self.dropping_proportions)

        fig.suptitle(f'k: {self.regions_dropped}, p: [{dropping_probabilities}]', fontweight="bold")
        fig.tight_layout()

        return axes
