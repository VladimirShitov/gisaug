from typing import Sized, Union, Optional, List

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


class ChangeAmplitude(Augmentation):
    """
    Augmentation, that randomly drops regions from time series to make it shorter in the time direction

    Parameters
    ----------
    c : float or tuple of 2 floats
        A multiplying coefficient for the amplitute. E.g., if c=0.8 means that the amplitude will become 20% smaller.
        If tuple of two numbers is provided (e.g. (`a`, `b`)), the coefficient will be chosen randomly and uniformly
        between `a` and `b` on each run.
    k : int or tuple with 2 integers
        Number of regions to augment. E.g. if c=0.8 and k=2, the amplitude of 2 regions will decrease by 20%. Note that
        the regions can overlap. If `c` is a tuple, new coefficient is generated for each region.
    region_start_fraction : float or tuple of 2 floats
        Proportion of the start of the augmented region to the length of the curve. E.g., if region_start_fraction=0.1,
        augmentation will not affect first 10% of the curve. If tuple of two numbers is provided (e.g. (`a`, `b`)),
        the proportion will be chosen randomly and uniformly between `a` and `b` for each region.
    region_end_fraction : float or tuple of 2 floats
        Proportion of the end of the augmented region to the length of the curve. E.g., if region_end_fraction=0.8,
        augmentation will not affect last 20% of the curve. If tuple of two numbers is provided (e.g. (`a`, `b`)),
        the proportion will be chosen randomly and uniformly between `a` and `b` for each region.
    clip : tuple of 2 floats (optional)
        Minimum and maximum possible value of the curve amplitude. E.g., if clip=(0, 1), the amplitude of the augmented
        curve will be clipped between 0 and 1.
    smooth : bool = False
        If True, change region amplitude smoothly, from a tiny change at the start of the region to
        the peak in its center
    """

    @staticmethod
    def normal_distribution_density(x, mean, sd):
        """Generate PDF of normal distribution with given parameters in the point `x`"""
        return np.exp(-(x - mean) ** 2 / (2 * sd ** 2)) / np.sqrt(np.pi * sd ** 2)

    @staticmethod
    def get_smooth_change(array: np.array, coef: float) -> np.array:
        """Smoothly increase/decrease `array` by `coef`

        array: np.array
            Time-series you want to change
        coef: float
            A number by which you want to multiply array. Changes will be equals to that number
            int the middle of `array` and slowly increasing/decreasing in the edges

        Returns
        -------
        Augmented `array`, which is smoothly multiplied by `coef`
        """
        difference = array * coef

        # Generate normal distribution for smooth change in amplitude
        mean = len(difference) / 2
        sd = len(difference) / 3
        normal_dist = np.array([
            ChangeAmplitude.normal_distribution_density(x, mean, sd) for x in range(len(difference))
        ])

        # Make it from 0 to 1
        normal_dist = (normal_dist - np.min(normal_dist)) / (np.max(normal_dist) - np.min(normal_dist))

        # Renormalize normal dist to a height of difference
        if coef < 1:
            normal_dist *= (1 - coef)  # Make it from 0 to (1 - coef)
            normal_dist = 1 - normal_dist  # Make it from 1 to coef
        else:
            normal_dist *= (coef - 1)  # Make it from 0 to (coef - 1)
            normal_dist = 1 + normal_dist  # Make it from 1 to coef

        array *= normal_dist

        return array

    @staticmethod
    def change_region_amplitude(
            x: np.array, coef: float, from_: int = 0, to: Optional[int] = None,
            clip: Optional[tuple] = None, smooth: bool = False
    ) -> np.array:
        """Multiply each point of `x` by `coef` between `from` and `to` indexes

        Parameters
        ----------
        x : numpy.array
            A numeric array with the timeseries to change
        coef : float
            A coefficient by which the interval of the `x` is multiplied. If absolute value of `coef` is less than 1,
            amplitude of the curve will decrease, otherwise it will increase. Be careful with negative `coef` (as it
            will mirror the curve) and with `clip` parameter
        from_ : int = 0
            Start of the interval of the curve to be multiplied by `coef`
        to : Optional[int] = None
            End of the interval (exclusive) of the curve to be multiplied by `coef`
        clip : Optional[tuple] = None
            If set, must be a tuple with 2 elements: minumum and maximum possible values of the amplitude. E.g. if
            curve values must be between 0 and 1, provide clip=(0, 1)
        smooth : bool = False
            If True, change region amplitude smoothly, from a tiny change at the start of the region to
            the peak in its center

        Returns
        -------
        numpy.array
            Augmented array with an interval multiplied by `coef`
        """
        if to is None:
            to = len(x)

        new_array = x.copy()

        if smooth:
            new_array[from_: to] = ChangeAmplitude.get_smooth_change(new_array[from_: to], coef)
        else:
            new_array[from_: to] *= coef

        if clip is not None:
            new_array = np.clip(new_array, clip[0], clip[1])

        return new_array

    def __init__(
           self, c: Union[float, Sized], k: Union[int, Sized], region_start_fraction: Union[float, Sized] = 0,
           region_end_fraction: Optional[Union[float, Sized]] = None, clip: Optional[tuple] = None, smooth: bool = False
    ):
        self.c = c
        self.k = k
        self.region_start_fraction = region_start_fraction
        self.region_end_fraction = region_end_fraction or 1
        self.clip = clip
        self.smooth = smooth

        self.multiplying_coefs = None
        self.n_regions = None
        self.region_starts = None
        self.region_ends = None
        self.should_generate_coef = None
        self.should_generate_k = None
        self.should_generate_start = None
        self.should_generate_end = None

        self._validate_parameters()
        self._generate_parameters()

    def _validate_parameters(self):
        # validate c
        try:
            is_valid_coefficient(self.c)
            self.should_generate_coef = False

        except (ValueError, TypeError):
            is_valid = are_valid_bounds(self.c)
            self.should_generate_coef = True

            if not is_valid:
                raise ValueError(f"{self.c} can't be validated as parameter c and is probably incorrect")

        # validate k
        try:
            is_valid_positive_integer(self.k)
            self.should_generate_k = False

        except (ValueError, TypeError):
            is_valid = are_valid_bounds(self.k)
            self.should_generate_k = True

            if not is_valid:
                raise ValueError(f"{self.k} can't be validated as a parameter k and is probably incorrect")

        # validate region_start_fraction
        try:
            # Though fraction is not probability, it has the same properties (i.e. float between 0 and 1)
            is_valid_probability(self.region_start_fraction)
            self.should_generate_start = False

        except (ValueError, TypeError):
            is_valid = are_valid_probability_bounds(self.region_start_fraction)
            self.should_generate_start = True

            if not is_valid:
                raise ValueError(f"{self.region_start_fraction} can't be validated as a parameter "
                                 f"region_start_fraction and is probably incorrect")

        # validate region_end_fraction
        try:
            is_valid_probability(self.region_end_fraction)
            self.should_generate_end = False

        except (ValueError, TypeError):
            is_valid = are_valid_probability_bounds(self.region_end_fraction)
            self.should_generate_end = True

            if not is_valid:
                raise ValueError(f"{self.region_end_fraction} can't be validated as a parameter "
                                 f"region_end_fraction and is probably incorrect")

        # Check that start is less than end or that their generation regions do not intersect
        if np.min(self.region_end_fraction) <= np.max(self.region_start_fraction):
            raise ValueError("Region start upper bound must be more than region end upper bound!")

        # validate clip
        if self.clip is not None:
            are_valid_bounds(self.clip)

    def _generate_parameters(self):
        """Generate parameters for augmentations. Requires `self.n_regions` to be set

        Sets
        ----
        self.multiplying_coefs
        self.n_regions
        self.region_starts
        self.region_ends
        """

        if self.should_generate_k:
            self.n_regions = np.random.randint(self.k[0], self.k[1])
        else:
            self.n_regions = self.k

        if self.should_generate_coef:
            self.multiplying_coefs = np.random.uniform(self.c[0], self.c[1], size=self.n_regions)
        else:
            self.multiplying_coefs = np.ones(shape=self.n_regions) * self.c

        if self.should_generate_start:
            # Generate number between 0 and lower bound of the end for each region
            self.region_starts = np.array(
                [np.random.uniform(0, np.min(self.region_end_fraction)) for _ in range(self.n_regions)])
        else:
            self.region_starts = np.array([self.region_start_fraction] * self.n_regions)

        if self.should_generate_end:
            self.region_ends = np.array(
                [np.random.uniform(start, 1) for start in self.region_starts])
        else:
            self.region_ends = np.array([self.region_end_fraction] * self.n_regions)

    def __call__(self, x: np.array) -> np.array:
        augmented_array = x.copy()

        for coef, region_start, region_end in zip(self.multiplying_coefs, self.region_starts, self.region_ends):
            start_idx = int(len(x) * region_start)
            end_idx = int(len(x) * region_end)

            augmented_array = self.change_region_amplitude(
                augmented_array, coef, from_=start_idx, to=end_idx, clip=self.clip, smooth=self.smooth)

        return augmented_array

    def visualize(self, x: np.array, vertical=True, figwidth=8, figheight=8):
        fig, axes = super().visualize(x, vertical, figwidth, figheight)

        fig.suptitle(f"k: {self.n_regions}, "
                     f"c: [{numbers_array_to_string(self.multiplying_coefs)}], "
                     f"region_start_fraction: [{numbers_array_to_string(self.region_starts)}], "
                     f"region_end_fraction: [{numbers_array_to_string(self.region_ends)}] "
                     f"smooth: {self.smooth}",
                     fontweight="bold")
        fig.tight_layout()

        return axes


class Pipeline(Augmentation):
    def __init__(self, augmentations: List[Augmentation], k: Optional[Union[int, Sized]]):
        self.augmentations = augmentations
        self.k = k or len(augmentations)  # If k is None, use all the augmentations

        self.n_augmentations = None
        self.should_generate_k = None
        self.used_augmentations = None

        self._validate_parameters()

    def _validate_parameters(self):
        try:
            is_valid_positive_integer(self.k)
            self.should_generate_k = False
            self.n_augmentations = self.k

        except (ValueError, TypeError):
            is_valid = are_valid_bounds(self.k)
            self.should_generate_k = True

            if not is_valid:
                raise ValueError(f"{self.k} can't be validated as a parameter k and is probably incorrect")

    def __call__(self, x: np.array):
        if self.should_generate_k:
            self.n_augmentations = np.random.randint(self.k[0], self.k[1])

        self.used_augmentations = []

        augmented_array = x.copy()

        for _ in range(self.n_augmentations):
            augmentation = np.random.choice(self.augmentations)
            self.used_augmentations.append(augmentation)

            augmented_array = augmentation(augmented_array)

        return augmented_array

    def visualize(self, x: np.array, vertical=True, figwidth=8, figheight=8):
        fig, axes = super().visualize(x, vertical=vertical, figwidth=figwidth, figheight=figheight)

        fig.suptitle(f"N augmentations: {self.n_augmentations}")
        fig.tight_layout()

        return axes
