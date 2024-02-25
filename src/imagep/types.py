"""Collects all types"""

# %%
from typing import Type, Union, Self
from pathlib import Path

import numpy as np

# > Local
from imagep.arrays.mdarray import mdarray

# %%
array = Union[
        np.ndarray,
        mdarray,
    ]


source_of_imgs = Union[
        str,
        Path,
        list[str | Path],
        array,
        list[array],
        Self,
    ]


indices = Union[
        int,
        list[int],
        tuple[int],
        slice,
        # list[slice],
        # np.ndarray,
    ]

