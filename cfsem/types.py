from typing import Union

from numpy import float64
from numpy.typing import NDArray

Array3xN = Union[NDArray[float64], tuple[NDArray[float64], NDArray[float64], NDArray[float64]]]
"""A 3xN array expressed as either a 2D numpy array or a tuple of 1D numpy arrays"""
