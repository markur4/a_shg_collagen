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
import imagep._utils.types as T
import imagep._configs.loggers as loggers
from imagep.images.mdarray import mdarray
import imagep.images._array_to_str as a2s


# %%
#
# ======================================================================
# == Class List Of (2D) arrays ===========================================
class l2Darrays:
    """An Alternative to np.ndarrays / ip.mdarray compatible with
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
        """Human readable representation of the object with information
        to reproduce the object.
        """
        return self._array_str(maximages=4)

    def __str__(self) -> str:
        """Human readable representation of the object."""
        return self._array_str()

    def _array_str(self, maximages: int = None) -> str:
        """Returns a string representation of the object"""

        rows = a2s.arrays_to_str(self.arrays, maximages=maximages).split("\n")

        rows[0] = (
            f"<class l2Darrays> with {len(self.arrays)} images of shapes {self.shapes}:"
        )

        return "\n".join(rows)

    #
    # == Shape Conversion ==============================================

    @staticmethod
    def _outerdim_to_list(
        input: T.array | list[T.array] | Self,
    ) -> list[T.array]:
        """Converts input into a list of 2D arrays:
        - Single 1D array is coerced into a 2D array and wrapped into a
          list
        - List of 1D arrays is returned as is
        - Single 2D array is wrapped into a list
        - List of 2D arrays is returned as is
        - Single 3D array is casted to a list
        - List of a single 3D array is retrieved and casted into list
        :param input: (list of) 1D, 2D, 3D array(s). List of 3D Arrays
           must contain only one 3D array. Will be converted into a list
           of 2D arrays
        :type input: np.array|mdarray|list[np.array|mdarray]
        :raises ValueError:
        - Arrays with dimensions >= 4D
        - List of arrays with dimensions >= 3D with more than 1 array
        - if a list of 1D or 3D arrays is passed, a ValueError is raised
        :return: list of 2D arrays
        :rtype: list[np.array|mdarray]
        """
        # > input = l2Darrays
        if isinstance(input, l2Darrays):
            return input.arrays
        elif isinstance(input, list):
            if isinstance(input[0], T.array):
                # > input = [np.array 1D]
                # ' or [np.array 1D, np.array 1D, ...]
                # ' or [np.array 2D, np.array 2D, ...]
                # if input[0].ndim == 1:
                #     # return [np.atleast_2d(arr) for arr in input]
                #     return input
                # elif input[0].ndim == 2:
                #     return input
                if input[0].ndim in (1, 2):
                    return input
                # > input = [np.array 3D]
                elif input[0].ndim == 3 and len(input) == 1:
                    return list(input[0])
                # > [np.array 3D, np.array 3D, ...]
                else:
                    raise ValueError(
                        f"List must contain 1D/2D arrays or a single 3D array"
                        f", not {input[0].ndim}D arrays"
                    )
            else:
                raise ValueError(
                    f"List must contain arrays, not {type(input[0])}"
                )
        # > input = np.array 1D/2D/3D
        elif isinstance(input, T.array):
            # if input.ndim == 1:
            #     # return [np.atleast_2d(input)]
            #     return [input]
            # elif input.ndim == 2:
            #     return [input]
            if input.ndim in (1, 2):
                return [input]
            elif input.ndim == 3:
                return list(input)
            else:
                raise ValueError(
                    f"When passing arrays, dimensions must be 1D/2D/3D, not {input.ndim}D array"
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

        shapes = [array.shape for array in self.arrays]
        maxdim = max([len(shape) for shape in shapes])
        ### Make a set of axis lengths for each axis from all shapes independently of dimensions
        shapes_sets = [
            set([shape[d] for shape in shapes if len(shape) > d])
            for d in range(maxdim)
        ]
        # y_set = {img.shape[0] for img in self.arrays}
        # x_set = {img.shape[1] for img in self.arrays}
        return (len(self.arrays), *shapes_sets)

    @property
    def shape(self):
        """Pops a shape from a random picture from the arrays
        and warns the user if there are multiple ones"""

        if not self.homogenous:  # and self._warnagain["shape"]:
            # loggers.DEBUG_LOGGER.warning(
            #     f"List contains {len(self.shapes[2])} different shapes,"
            #     " retrieving the shape of the first image. Try using"
            #     " .shapes to display unique shapes.",
            # )
            pass

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

    @property
    def dtypes_pretty(self) -> str:
        """Retrieves dtype from one of its elements."""
        strings = [str(dtype) for dtype in self.dtypes]
        return str(set(strings))  # Make set to keep the curled brackets

    def astype(self, dtype: np.dtype):
        return l2Darrays([img.astype(dtype) for img in self.arrays])

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
            return l2Darrays(self.arrays[val])
        # > self[1,2,3] or self[[1,2,3]]
        elif isinstance(val, (tuple, list)):
            return l2Darrays([self.arrays[v] for v in val])
        # > self[ array([True, False, ...]) ]
        elif isinstance(val, (T.array, l2Darrays)):
            # > Since numpy intentionally loses dimension information, we might
            # ' also abandon metadata information
            return np.concatenate(
                [array[val[i]] for i, array in enumerate(self.arrays)]
            )
            # > This here DOES paritally preserve shape (z, but not y)
            # return [array[val[i]] for i, array in enumerate(self.arrays)]

        else:
            raise ValueError(f"Invalid indexer '{val}'")

    def __setitem__(
        self, val: int | slice | tuple | list, item: T.array | list[T.array]
    ):
        # > self[boolean_index] = int
        if isinstance(val, (T.array, l2Darrays)) and np.isscalar(item):
            for i, array in enumerate(self.arrays):
                array[val[i]] = item

            return  #!! Return early

        ### > Convert scalar item to list of arrays
        if isinstance(item, (int, float)) or np.isscalar(item):
            # > self[int:int:int] = int
            if isinstance(val, slice):
                indices = range(*val.indices(len(self.arrays)))
                item = [np.full_like(self.arrays[i], item) for i in indices]
            # > self[int, int, int] = int
            elif isinstance(val, (tuple, list)):
                item = [np.full_like(self.arrays[i], item) for i in val]
            # > self[int] = int
            else:
                item = [np.full_like(self.arrays[val], item)]

        ### item is now an array
        # > Set to list of 2D arrays if not already
        item: list[T.array] = self._outerdim_to_list(item)

        # > self[int] = [np.array]
        if isinstance(val, int) and len(item) == 1:
            self.arrays[val] = item[0]
        # > self[int] = [np.array, np.array]
        elif isinstance(val, int) and len(item) > 1:
            raise ValueError(f"Can't set {len(item)} arrays to a single index")
        # > self[1:10:2] = [np.array] !! Shapes must be compatible
        elif isinstance(val, slice):
            self._set_sequence_into_sliceindex(val, item)
        # > self[1,2,3] = [np.array]  or self[[1,2,3]] = [np.array]
        elif isinstance(val, (tuple, list)):
            self._set_sequence_into_listindex(val, item)
        else:
            raise ValueError(f"Invalid indexer '{val}'")

    def _set_sequence_into_sliceindex(
        self,
        val: slice,
        item: T.array | list[T.array],
    ):
        """Sets a sequence of arrays into a slice of self (that's
        compatible with the shape of that slice).
        e.g. self[1:10:2] = [np.array]
        """
        indices = range(*val.indices(len(self.arrays)))
        insertlength = len(indices)
        if len(item) == 1:  # > self[1:10:2] = [np.array]
            # > Multiple entries are being overwritten by a single array
            item = item * insertlength
        elif len(item) != insertlength:  # > self[1:10:2] = [np.array, ...]
            raise ValueError(
                f"Can't set {len(item)} items to a slice of length {insertlength}, their lengths must match."
            )
        ### Insert into slice (with regards to stepsize)
        self.arrays = l2Darrays(
            [
                (self.arrays[i] if i not in indices else item[indices.index(i)])
                for i in range(len(self.arrays))
            ]
        )

    def _set_sequence_into_listindex(
        self,
        val: list | tuple,
        item: T.array | list[T.array],
    ):
        """Sets a sequence of arrays into a list-slice of self (that's
        compatible with the shape of that slice).
        e.g. self[1,2,3] = [np.array]
        """
        insertlength = len(val)
        if len(item) == 1:  # > self[1,2,3] = [np.array]
            # > Multiple entries are being overwritten by a single array
            item = item * insertlength
        elif len(item) != insertlength:  # > self[1,2,3] = [np.array, ...]
            raise ValueError(
                f"Can't set {len(item)} items to a slice of length {insertlength}, their lengths must match."
            )
        self.arrays = l2Darrays(
            [
                self.arrays[i] if i not in val else item[val.index(i)]
                for i in range(len(self.arrays))
            ]
        )

    #
    # == Copy & Conversions ============================================

    def copy(self) -> Self:
        """Returns a copy of the object"""
        return l2Darrays([img.copy() for img in self.arrays])

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

    def max(self, **kws) -> int | float:
        if self.homogenous:
            return self.asarray().max(**kws)
        else:
            #!! mdarray.max() returns an 0D array..?
            return max([img.max(**kws).item() for img in self.arrays])

    def min(self, **kws) -> int | float:
        if self.homogenous:
            return self.asarray().min(**kws)
        else:
            return min([img.min(**kws).item() for img in self.arrays])

    # ==================================================================
    # == OPERATIONS ====================================================

    #
    # === Main ===
    def _operation(
        self,
        other: int | float | np.dtype | T.array | Self,
        operation: Callable,
    ) -> Self:

        ### Check Type
        isscalar: bool = np.isscalar(other) or isinstance(other, (int, float))
        if not (isscalar or isinstance(other, (T.array, l2Darrays))):
            raise TypeError(
                f"Math operations are possible only with a scalar or a (list of) array(s), not {type(other)}"
            )
        # > self + 2
        elif isscalar:
            return l2Darrays([operation(img, other) for img in self.arrays])
        # > or self + np.array (1D or 2D)
        # > or self + l2Darrays (1D or 2D)
        elif other.ndim in (1, 2):
            return l2Darrays([operation(img, other) for img in self.arrays])
        # > self + np.array (3D)
        elif other.ndim == 3 and len(other) == len(self.arrays):
            return l2Darrays(
                [operation(img, other[i]) for i, img in enumerate(self.arrays)]
            )
        else:
            raise ValueError(
                f"Can't perform operations between list2Darray with"
                f" shapes {self.shapes} with an array of shape {other.shape}"
            )

    #
    # === +, -, *, / ===
    def __add__(self, other):
        operation = lambda x1, x2: x1 + x2
        return self._operation(other, operation)

    def __sub__(self, other):
        operation = lambda x1, x2: x1 - x2
        return self._operation(other, operation)

    def __mul__(self, other):
        operation = lambda x1, x2: x1 * x2
        return self._operation(other, operation)

    def __truediv__(self, other):
        # if other == 0:
        #     raise ZeroDivisionError("Can't divide by zero")
        operation = lambda x1, x2: x1 / x2
        return self._operation(other, operation)

    # === //, %, ** ===
    def __floordiv__(self, other):
        operation = lambda x1, x2: x1 // x2
        return self._operation(other, operation)

    def __mod__(self, other):
        operation = lambda x1, x2: x1 % x2
        return self._operation(other, operation)

    def __pow__(self, other):
        operation = lambda x1, x2: x1**x2
        return self._operation(other, operation)

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
    # === ==, !=, <, <=, >, >= ===
    def __eq__(self, other):
        operation = lambda x1, x2: x1 == x2
        return self._operation(other, operation)

    def __ne__(self, other):
        operation = lambda x1, x2: x1 != x2
        return self._operation(other, operation)

    def __lt__(self, other):
        operation = lambda x1, x2: x1 < x2
        return self._operation(other, operation)

    def __le__(self, other):
        operation = lambda x1, x2: x1 <= x2
        return self._operation(other, operation)

    def __gt__(self, other):
        operation = lambda x1, x2: x1 > x2
        return self._operation(other, operation)

    def __ge__(self, other):
        operation = lambda x1, x2: x1 >= x2
        return self._operation(other, operation)

    #
    # !! == End  Class =================================================


# %%
### Test as part of Collection
if __name__ == "__main__":
    import imagep as ip

    folder = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = ip.Stack(
        data=folder,
        fname_extension="txt",
        verbose=True,
        pixel_length=(1.5 * 115.4) / 1024,
        imgname_position=1,
    )
    I = 6

    # %%
    larry = l2Darrays(arrays=list(Z.imgs))
    larry.shapes
    # Z.imgs.tolist()

    # %%
    larry_hetero = l2Darrays(
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
