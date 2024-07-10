from typing import Tuple, Union
from numpy.typing import NDArray
from numpy import float64

Array3xN = Union[
    NDArray[float64], Tuple[NDArray[float64], NDArray[float64], NDArray[float64]]
]
