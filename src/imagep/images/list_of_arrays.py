"""A Class to store arrays of different sizes (and types) by replacing
the outermost dimension of the image stacks with a list."""

# %%+
from typing import Self

import numpy as np

from imagep.images.mdarray import mdarray


#
# ======================================================================
# == Class ListOfArrays ================================================
class ListOfArrays:
    """An Alternative to np.ndarrays / ip.Mdarray compatible with
    storing images of different sizes and types. Provides a
    np.array-esque experience, giving access to properties like shape and
    dtype without complicating subsequent code.
    """

    def __init__(self, arrays: list[mdarray | np.ndarray]):
        if not isinstance(arrays, list) and isinstance(arrays[0], np.ndarray):
            raise TypeError(
                f"Must pass list of numpy arrays, not '{type(arrays)}'"
            )

        ### larrys = list of Arrays
        self.arrays: list[mdarray | np.ndarray] = arrays

    #
    # == Shape =========================================================
    @property
    def shape(self):
        """Returns (z, {y}, {x}), with x and y sets of all widths and
        heights occurring in the data"""
        y = {img.shape[0] for img in self.arrays}
        x = {img.shape[1] for img in self.arrays}
        return (len(self.arrays), y, x)

    def __len__(self):
        return len(self.arrays)

    #
    # == Set and Get items =============================================

    def __getitem__(self, value: int | tuple | list | slice):
        if isinstance(value, (tuple, list)):
            if all(isinstance(v, int) for v in value):
                return [self.arrays[v] for v in value]
            elif all(isinstance(v, (list, slice)) for v in value):
                return [self.arrays[v] for v in value]
            else:
                raise ValueError(f"Invalid indexer '{value}'")
        elif isinstance(value, (slice, int)):
            return self.arrays[value]
        else:
            raise ValueError(f"Invalid indexer '{value}'")

    def __setitem__(self, value: int | tuple | list | slice, item: mdarray):
        if isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                self.arrays[v] = item[i]
        elif isinstance(value, (slice, int)):
            self.arrays[value] = item
        else:
            raise ValueError(f"Invalid indexer '{value}'")

    #
    # == dtype =========================================================

    @property
    def dtype(self):
        return self.arrays[0].dtype if self.arrays else None

    def astype(self, dtype: np.dtype):
        return [img.astype(dtype) for img in self.arrays]

    #
    # == Copy & Conversions ============================================

    def copy(self) -> Self:
        """Returns a copy of the object"""
        return ListOfArrays([img.copy() for img in self.arrays])
        # return [img.copy() for img in self.larry]

    @property
    def asarray(self):
        """Returns a 3D numpy array. Metadata gets lost !!!"""
        return np.array(self.arrays, dtype=self.dtype)

    #
    # == Statistics ====================================================

    def max(self, **kws):
        return self.asarray.max(**kws)

    def min(self, **kws):
        return self.asarray.min(**kws)

    #
    # !! == End  Class =================================================


if __name__ == "__main__":
    import imagep as ip

    folder = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = ip.Collection(
        data=folder, fname_extension="txt", verbose=True, x_µm=1.5 * 115.4
    )
    I = 6

    # %%
    loar = ListOfArrays(arrays=list(Z.imgs))
    loar.shape
    # Z.imgs.tolist()
