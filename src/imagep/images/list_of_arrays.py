"""A Class to store arrays of different sizes (and types) by replacing
the outermost dimension of the image stacks with a dynamic list.
(This isn't PascalCase because it's a list.)
"""

# %%
from typing import Self, Type
import copy

import logging

import numpy as np

import warnings

# > Local
import imagep._utils.types as T
from imagep.images.mdarray import mdarray


#
# ======================================================================
# == Class List Of 2D arrays ===========================================
class list2Darrays:
    """An Alternative to np.ndarrays / ip.Mdarray compatible with
    storing images of different sizes and types. Provides a
    np.array-esque experience, giving access to properties like shape and
    dtype without complicating subsequent code.
    """

    def __init__(self, arrays: T.array | list[T.array], dtype=None):

        ### Convert all arrays to the specified dtype
        if dtype is not None:
            arrays = [arr.astype(dtype) for arr in arrays]

        ### Set to list of 2D arrays if not already
        self.arrays: list[T.array] = self._outerdim_to_list(arrays)

        self._logger = logging.getLogger(__name__)

    #
    # == Shape =========================================================

    @staticmethod
    def _outerdim_to_list(input: T.array | list[T.array]) -> list[T.array]:
        """Converts input into a list of 2D arrays
        - If a single 2D array is passed, it's wrapped in a list
        - If a list of 2D arrays is passed, it's returned as is
        - If a 3D array is passed, it's converted into a list of 2D
          arrays
        - if a 1D or 4D array o is passed, a ValueError is raised
        - if a list of 1D or 3D arrays is passed, a ValueError is raised
        """

        if isinstance(input, list):
            if isinstance(input[0], T.array):
                if input[0].ndim == 2:
                    return input
                else:
                    raise ValueError(
                        f"List must contain 2D arrays, not {input[0].ndim}D arrays"
                    )
            else:
                raise ValueError(
                    f"List must contain arrays, not {type(input[0])}"
                )
        elif isinstance(input, T.array):
            if input.ndim == 2:
                return [input]
            elif input.ndim == 3:
                return list(input)
            else:
                raise ValueError(
                    f"Must pass a 2D or 3D array, not {input.ndim}D array"
                )

        else:
            raise ValueError(
                f"Must pass a 2D or 3D array or a list of 2D arrays, not '{type(input)}'"
            )

    @property
    def shapes(self):
        """Returns (z, {y}, {x}), with x and y sets of all widths and
        heights occurring in the data"""
        y_set = {img.shape[0] for img in self.arrays}
        x_set = {img.shape[1] for img in self.arrays}
        return (len(self.arrays), y_set, x_set)

    @property
    def shape(self):
        """Pops a shape from a random picture from the arrays
        and warns the user if there are multiple ones"""

        ### Warn if more than one shape
        if len(self.shapes[1]) > 1 or len(self.shapes[2]) > 1:
            warnings.warn(
                f"List contains {self.shapes[1]} different shapes,"
                " retrieving the shape of the first image",
                stacklevel=99,
            )
        shapes = [img.shape for img in self.arrays]
        return (len(self.arrays), *shapes[0])

    def __len__(self):
        return len(self.arrays)

    #
    # == Set and Get items =============================================

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, val: T.indices) -> T.array | Self:
        """Slice the list of arrays.
        - Does NOT preserve dimensions. That means that the
        result can be either an array or a list, but if list, it's a
        ListOfArrays to mimic numpy behavior.

        """
        # > self[1] or self[1:3]
        if isinstance(val, int):
            return self.arrays[val]
        # > self[1:3]
        if isinstance(val, slice):
            return list2Darrays(self.arrays[val])
        # > self[1,2,3] or self[[1,2,3]]
        elif isinstance(val, (tuple, list)):
            return list2Darrays([self.arrays[v] for v in val])
        else:
            raise ValueError(f"Invalid indexer '{val}'")

    def __setitem__(
        self, val: int | slice | tuple | list, item: T.array | list[T.array]
    ):

        ### Set to list of 2D arrays if not already
        item: list[T.array] = self._outerdim_to_list(item)

        # > self[int] = [np.array]
        if isinstance(val, int) and len(item) == 1:
            self.arrays[val] = item[0]
        # > self[int] = [np.array, np.array]
        elif isinstance(val, int) and len(item) > 1:
            raise ValueError(f"Can't set {len(item)} arrays to a single index")
        # > self[1:10:2] = [np.array]
        elif isinstance(val, slice):
            indices = range(*val.indices(len(self.arrays)))
            insertlength = len(indices)
            if len(item) == 1:  # > self[1:10:2] = [np.array]
                # > Multiple entries are being overwritten by a single array
                item = item * insertlength
            elif len(item) != insertlength:  # > self[1:10:2] = [np.array, ...]
                raise ValueError(
                    f"Can't set {len(item)} items to a slice of length {insertlength}, their lengths must match."
                )
            ### insert with regards to stepsize
            self.arrays = list2Darrays(
                [
                    (
                        self.arrays[i]
                        if i not in indices
                        else item[indices.index(i)]
                    )
                    for i in range(len(self.arrays))
                ]
            )
        # > self[1,2,3] = [np.array]
        elif isinstance(val, (tuple, list)):
            insertlength = len(val)
            if len(item) == 1:  # > self[1,2,3] = [np.array]
                # > Multiple entries are being overwritten by a single array
                item = item * insertlength
            elif len(item) != insertlength:  # > self[1,2,3] = [np.array, ...]
                raise ValueError(
                    f"Can't set {len(item)} items to a slice of length {insertlength}, their lengths must match."
                )
            self.arrays = list2Darrays(
                [
                    self.arrays[i] if i not in val else item[val.index(i)]
                    for i in range(len(self.arrays))
                ]
            )
        else:
            raise ValueError(f"Invalid indexer '{val}'")

    #
    # == dtype =========================================================

    @property
    def dtypes(self) -> set[Type]:
        """Retrieves dtype from one of its elements."""
        return {img.dtype for img in self.arrays}

    def astype(self, dtype: np.dtype):
        return list2Darrays(self.arrays, dtype=dtype)

    @property
    def dtype(self) -> Type:
        """Retrieves the datatype with biggest bit depth"""

        get_bytes = lambda x: np.dtype(x).itemsize  # > x = np.float64
        max_dtype = max(self.dtypes, key=get_bytes)
        if len(self.dtypes) > 1:
            self._logger.warning(
                f"List contains {len(self.dtypes)} different dtypes,"
                f" retrieving the dtype with largest bitdepth: {max_dtype}."
                " To remove this warning, homogenize the arrays"
                " using .astype()",
                stacklevel=99,
            )
        return max_dtype

    #
    # == Copy & Conversions ============================================

    def copy(self) -> Self:
        """Returns a copy of the object"""
        return list2Darrays([img.copy() for img in self.arrays])

    def asarray(self, dtype=None):
        """Returns a 3D numpy array. This has these consequences:
        - Metadata gets lost
        - Inhomogenous dtypes are converted into the largest dtype
        """
        if len(self.shapes[1]) != 1 and len(self.shapes[2]) != 1:
            raise ValueError(
                f"Can't convert list of inhomogenous arrays into an array [(z,y,x) = {self.shapes}]"
            )
        dtype = self.dtype if dtype is None else dtype

        return np.array(self.arrays, dtype=dtype)

    #
    # == Statistics ====================================================

    def max(self, **kws):
        return self.asarray.max(**kws)

    def min(self, **kws):
        return self.asarray.min(**kws)

    #
    # !! == End  Class =================================================


# %%
### Test as part of Collection
if __name__ == "__main__":
    import imagep as ip

    folder = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = ip.Collection(
        data=folder,
        fname_extension="txt",
        verbose=True,
        pixel_length=(1.5 * 115.4) / 1024,
        imgname_position=1,
    )
    I = 6

    # %%
    loar = list2Darrays(arrays=list(Z.imgs))
    loar.shapes
    # Z.imgs.tolist()
