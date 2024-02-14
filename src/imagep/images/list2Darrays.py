"""A Class to store arrays of different sizes (and types) by replacing
the outermost dimension of the image stacks with a dynamic list.
(This isn't PascalCase because it's a list.)
"""

# %%

import re

from typing import Self, Type, Callable
import copy

# from collections import defaultdict

import numpy as np

# import logging

# import warnings

# > Local
import imagep._utils.utils as ut
import imagep._configs.rc as rc
import imagep._configs.loggers as loggers
import imagep._utils.types as T
from imagep.images.mdarray import mdarray
from imagep.images.array_to_str import array3D_to_str


# %%
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

    #
    # == Representation ================================================
    def __repr__(self) -> str:
        return self._array_str(maximages=4)

    def __str__(self) -> str:
        return self._array_str()

    def _array_str(self, maximages: int = None) -> str:
        """Returns a string representation of the object"""

        rows = array3D_to_str(self.arrays, maximages=maximages).split("\n")

        rows[0] = f"<class list2Darrays> with {len(self.arrays)} images:"

        return "\n".join(rows)

    #
    # == Shape Conversion ==============================================

    @staticmethod
    def _outerdim_to_list(
        input: T.array | list[T.array] | Self,
    ) -> list[T.array]:
        """Converts input into a list of 2D arrays
        - If a single 2D array is passed, it's wrapped in a list
        - If a list of 2D arrays is passed, it's returned as is
        - If a 3D array is passed, it's converted into a list of 2D
          arrays
        - if a 1D or 4D array o is passed, a ValueError is raised
        - if a list of 1D or 3D arrays is passed, a ValueError is raised
        """
        if isinstance(input, list2Darrays):
            return input.arrays
        elif isinstance(input, list):
            if isinstance(input[0], T.array):
                # > [np.array 2D, np.array 2D, ...]
                if input[0].ndim == 2:
                    return input
                # > [np.array 3D]
                elif input[0].ndim == 3 and len(input) == 1:
                    return list(input[0])
                # > [np.array 3D, np.array 3D, ...]
                else:
                    raise ValueError(
                        f"List must contain 2D arrays, not {input[0].ndim}D arrays"
                    )
            else:
                raise ValueError(
                    f"List must contain arrays, not {type(input[0])}"
                )
        elif isinstance(input, T.array):
            # > np.array 2D
            if input.ndim == 2:
                return [input]
            # > np.array 3D
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

    #
    # == Shape Properties ==============================================

    def __len__(self):
        return len(self.arrays)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def homogenous(self):
        return len(self.shapes[1]) == 1 or len(self.shapes[2]) == 1

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

        if not self.homogenous:  # and self._warnagain["shape"]:
            loggers.DEBUG_LOGGER.warning(
                f"List contains {len(self.shapes[2])} different shapes,"
                " retrieving the shape of the first image. Try using"
                " .shapes to display unique shapes.",
            )

        shapes = [img.shape for img in self.arrays]

        return (len(self.arrays), *shapes[0])

    #
    # == dtype =========================================================

    def _warn_dtype_inhomogenous(self, start_msg: str = ""):

        ### Message
        M = (
            f"List contains {len(self.dtypes)} different dtypes."
            f" We use the dtype with largest bitdepth: {self.dtype_maxbits}."
            " (To remove this warning, homogenize the arrays"
            " using .astype())"
        )
        M = f"{start_msg} " + M if start_msg else M

        ### Log and show warning
        loggers.DEBUG_LOGGER.warning(
            M,
            #  stacklevel=99
        )

    @property
    def dtypes(self) -> set[Type]:
        """Retrieves dtype from one of its elements."""
        return {img.dtype for img in self.arrays}

    def astype(self, dtype: np.dtype):
        return list2Darrays(self.arrays, dtype=dtype)

    @property
    def dtype(self) -> Type:
        """Retrieves the datatype with biggest bit depth"""
        if not self.homo_types:
            self._warn_dtype_inhomogenous("Calling dtype.")
        return self.dtype_maxbits

    @property
    def dtype_maxbits(self):
        get_bytes = lambda x: np.dtype(x).itemsize  # > x = np.float64
        return max(self.dtypes, key=get_bytes)

    @property
    def homo_types(self) -> bool:
        if len(self.dtypes) == 1:
            return True
        else:
            return False

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
    # == Copy & Conversions ============================================

    def copy(self) -> Self:
        """Returns a copy of the object"""
        return list2Darrays([img.copy() for img in self.arrays])

    def asarray(self, dtype=None):
        """Returns a 3D numpy array. This has these consequences:
        - !!! Metadata gets lost
        - Inhomogenous dtypes are converted into the largest dtype
        """
        if not self.homo_types:
            self._warn_dtype_inhomogenous(start_msg="Using .asarray().")

        if not self.homogenous:
            raise ValueError(
                f"Can't convert list of inhomogenous arrays into an array [(z,y,x) = {self.shapes}]"
            )
        dtype = self.dtype if dtype is None else dtype

        # if as_numpy:
        # !! returning as mdarray does NOT preserve metadata, too!
        return np.array(self.arrays, dtype=dtype)
        # else:
        #     # !! Return as a 3D mdarray
        #     return mdarray(self.arrays, dtype=dtype)

    #
    # == Statistics ====================================================

    def max(self, **kws):
        if self.homogenous:
            return self.asarray().max(**kws)
        else:
            return max([img.max(**kws) for img in self.arrays])

    def min(self, **kws):
        if self.homogenous:
            return self.asarray().min(**kws)
        else:
            return min([img.min(**kws) for img in self.arrays])

    #
    # == Math Operations ===============================================

    def _math_operation(
        self,
        other: int | float | np.dtype | T.array | Self,
        operation: Callable,
    ) -> Self:

        ### Check Type
        isscalar: bool = np.isscalar(other) or isinstance(other, (int, float))
        if not (isscalar or isinstance(other, (T.array, list2Darrays))):
            raise TypeError(
                f"Math operations are possible only with a scalar or a (list of) array(s), not {type(other)}"
            )
        # > self + 2
        elif isscalar:
            return list2Darrays([operation(img, other) for img in self.arrays])
        # > or self + np.array (1D or 2D)
        # > or self + list2Darrays (1D or 2D)
        elif other.ndim in (1, 2):
            return list2Darrays([operation(img, other) for img in self.arrays])
        # > self + np.array (3D)
        elif other.ndim == 3 and len(other) == len(self.arrays):
            return list2Darrays(
                [operation(img, other[i]) for i, img in enumerate(self.arrays)]
            )
        else:
            raise ValueError(
                f"Can't perform math operations between list2Darray with"
                f" shapes {self.shapes} with an array of shape {other.shape}"
            )

    def __add__(self, other):
        operation = lambda x1, x2: x1 + x2
        return self._math_operation(other, operation)

    def __sub__(self, other):
        operation = lambda x1, x2: x1 - x2
        return self._math_operation(other, operation)

    def __mul__(self, other):
        operation = lambda x1, x2: x1 * x2
        return self._math_operation(other, operation)

    def __truediv__(self, other):
        # if other == 0:
        #     raise ZeroDivisionError("Can't divide by zero")
        operation = lambda x1, x2: x1 / x2
        return self._math_operation(other, operation)

    def __floordiv__(self, other):
        operation = lambda x1, x2: x1 // x2
        return self._math_operation(other, operation)

    def __mod__(self, other):
        operation = lambda x1, x2: x1 % x2
        return self._math_operation(other, operation)

    def __pow__(self, other):
        operation = lambda x1, x2: x1**x2
        return self._math_operation(other, operation)

    # def __lshift__(self, other):
    #     operation = lambda x1, x2: x1 << x2
    #     return self._math_operation(other, operation)

    # def __rshift__(self, other):
    #     operation = lambda x1, x2: x1 >> x2
    #     return self._math_operation(other, operation)

    # def __and__(self, other):
    #     operation = lambda x1, x2: x1 & x2
    #     return self._math_operation(other, operation)

    # def __xor__(self, other):
    #     operation = lambda x1, x2: x1 ^ x2
    #     return self._math_operation(other, operation)

    # def __or__(self, other):
    #     operation = lambda x1, x2: x1 | x2
    #     return self._math_operation(other, operation)

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
    larry = list2Darrays(arrays=list(Z.imgs))
    larry.shapes
    # Z.imgs.tolist()

    # %%
    larry_hetero = list2Darrays(
        arrays=[
            np.ones((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float16) * 2.5,
            np.ones((4, 4), dtype=np.float32) * 3.5,
            np.ones((5, 5), dtype=np.uint8),
            np.ones((10, 10), dtype=np.uint8),
            np.ones((15, 15), dtype=np.uint8) * 255,
            np.ones((20, 20), dtype=np.float64),
            np.ones((30, 30), dtype=np.float16),
            np.ones((1024, 1024), dtype=np.float32),
        ],
    )
    ### Make numbers heterogenous
    for z, img in enumerate(larry_hetero):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                img[y, x] = img[y, x] / (x + 1) / (y + 1) * (z + 1)

    # %%
    ### Test __repr__  for numpy array
    larry_hetero

    # %%
    ### Add metadata
    for z, img in enumerate(larry_hetero):
        larry_hetero[z] = mdarray(
            img,
            # pixel_length=20,
            name=f"testImage {z}",
        )

    # %%
    print(larry_hetero.dtype)
    print(larry_hetero.shape)

    # %%
    ### Test __repr__  for mdarray
    larry_hetero
    # %%
    ### Test print()
    print(larry_hetero)

    # %%
    ### Get full info
    larry_hetero._array_str()
