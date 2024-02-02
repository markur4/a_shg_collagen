"""Classes to handle raw image formats and slicing"""
# %%
from typing import Self, TYPE_CHECKING

import copy

from pathlib import Path
import numpy as np

# > Local
import imagep._rc as rc
import imagep.images.importtools as importtools

# from imagep.images.imgs import Imgs

if TYPE_CHECKING:
    from imagep.images.imgs import Imgs


# %%
# == Class ImgsImport =====================================================
class ImgsImport:
    """Class for handling the most basic functionalities:
    - Imports of raw image data
    - Slicing of image stacks
    """

    PATH_PLACEHOLDER = "Source: Numpy array"

    def __init__(
        self,
        data: str | Path | np.ndarray | list[np.ndarray] | Self = None,
        dtype: np.dtype = rc.DTYPE_DEFAULT,
        verbose: bool = True,
    ) -> None:
        ### Make sure that either path or array is given
        self.verbose = verbose

        ### Init attributes, they will be set by the import functions
        self.path: str | Path = None
        self.imgs: np.ndarray = None

        ### Remember the dtype that arrays will be converted in
        # ?? Should we keep original dtypes or explicitly convert to this?
        self._dtype = dtype

        ### Process source into path and images
        self._source_type = type(data)
        self._import(data)

        ### Slicing
        # > Remember if this object has been sliced
        self._slice: bool | str = False
        self._num_imgs: int = self.imgs.shape[0]
        self._slice_indices: list[int] = list(range(self._num_imgs))

    #
    # == Process Source Data ===========================================

    def _check_data_type(
        self, data: str | Path | np.ndarray | list[np.ndarray] | Self
    ) -> None:
        """Check if data is a valid image source"""
        types = (str, Path, np.ndarray, list, type(self))
        m = (
            " Either path, array or an instance of Imgs"
            + " (or Imgs-subclasses) must be given."
        )
        if data is None:
            raise ValueError("No image source passed." + m)
        elif not isinstance(data, types):
            raise ValueError(f"Unknown type of image source: {type(data)}." + m)

    def _import(
        self, data: str | Path | np.ndarray | list[np.ndarray] | Self
    ) -> None:
        self._check_data_type(data)

        if isinstance(data, (str, Path)):
            self.from_path(data)
        elif isinstance(data, (np.ndarray, list)):
            self.from_array(data)
        # !! Importing from Instance is replaced by full attribute transfer
        # > Keep this here for future
        elif isinstance(data, type(self)):
            self.from_instance(data)

    #

    #
    # == From Path, Array or Instance ==================================
    def from_path(self, path: str | Path) -> None:
        """Import images from a folder"""

        ### If string, convert to Path
        path = Path(path)

        ### Check if path is valid
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        ### Set Path
        self.path = Path(path)

        ### Import
        if self.verbose:
            print(f"=> Importing images from '{self.path_short}' ...")
        self.imgs = self.import_imgs(path, dtype=self._dtype)

        if self.verbose:
            print(
                f"   Import DONE ({self.imgs.shape[0]} images,"
                f" {self.imgs.shape[1]}x{self.imgs.shape[2]},"
                f" {self.imgs.dtype})"
            )
            print()

    def from_array(self, array: np.ndarray | list) -> None:
        """Import images from a numpy array"""
        ### If list of arrays
        waslist = False
        if isinstance(array, list):
            waslist = True
            array = np.array(array)

        ### Check for shape (z, y, x)
        if len(array.shape) != 3:
            raise ValueError(
                f"Array must have shape (z, y, x), not {array.shape}"
            )
        ### Set path and imgs
        if self.verbose:
            m = " list of arrays" if waslist else " array"
            print(
                f"=> Transferring"
                + m
                + f" ({array.shape[0]} images "
                + f" {array.shape[1]}x{array.shape[2]},"
                + f" {array.dtype}) ..."
            )
        self.path = self.PATH_PLACEHOLDER
        self.imgs = array.astype(self._dtype)

        if self.verbose:
            print("   Transfer DONE")
            print()

    def from_instance(self, instance: Self, verbose: bool) -> None:
        """Transfer images and path from another instance"""

        # !! This mustn't be self.verbose, because it's not set yet
        if verbose:
            print(
                f"=> Transferring attributes from an instance"
                + f" ({instance.imgs.shape[0]} images "
                + f" {instance.imgs.shape[1]}x{instance.imgs.shape[2]},"
                + f" {instance.imgs.dtype}; "
                + f" from: '{instance.path_short}') ..."
            )
        ### Full atrribute transfer
        for attr, value in instance.__dict__.items():
            setattr(self, attr, value)

        if verbose:
            print("   Transfer DONE")
            print()

    #
    # == Path ==========================================================

    @property
    def path_short(self) -> str:
        """Shortened path"""
        if self.path == self.PATH_PLACEHOLDER:
            return self.PATH_PLACEHOLDER
        else:
            return str(self.path.parent.name + "/" + self.path.name)

    #
    # == Import From Files =============================================

    @staticmethod
    def import_imgs(path: Path, dtype: np.dtype) -> np.ndarray:
        """Import z-stack from a folder"""

        ### Get all txt files
        txts = list(path.glob("*.txt"))

        ### sort txts by number
        txts = sorted(txts, key=lambda x: int(x.stem.split("_")[-1]))

        ### Invert, since the first image is the bottom one
        txts = txts[::-1]

        ### Import all txt files
        imgs = []
        for txt in txts:
            imgs.append(importtools.from_txt(txt))

        ### Convert to numpy array
        imgs = np.array(imgs, dtype=dtype)

        return imgs

    #
    # == Access/Slice Images ===========================================

    @property
    def stack_raw(self) -> np.ndarray:
        return self.import_imgs()

    def __iter__(self):
        return iter(self.imgs)

    def __getitem__(self, val: slice) -> Self | "Imgs":
        # > Create a copy of this instance
        _self = copy.deepcopy(self)
        # > Assign the sliced imgs to the new instance
        # indices = ut.indices_from_slice(slice=val, n_imgs=self.imgs.shape[0])

        ### Slice while preserving dimension information
        # > Z[0]
        if isinstance(val, int):
            _self.imgs = self.imgs[[val], ...]
            indices = [val]
        # > Z[1:3]
        elif isinstance(val, slice):
            _self.imgs = self.imgs[val, ...]
            indices = range(*val.indices(self._num_imgs))
        # > Z[1,2,5]
        elif isinstance(val, tuple):
            _self.imgs = self.imgs[list(val), ...]
            indices = val
        # > or Z[[1,2,5]] pick multiple images
        elif isinstance(val, list):
            _self.imgs = self.imgs[val, ...]
            indices = val

        ### Remember how this object was sliced
        _self._slice = str(val)
        _self._slice_indices = indices

        return _self

    #
    # == I/O ===========================================================

    def save_as_nparray(self, fname: str | Path) -> None:
        """Save the z-stack to a folder"""
        np.save(fname, self.imgs)

    def load_from_nparray(self, fname: str | Path) -> None:
        """Load the z-stack from a folder"""
        self.imgs = np.load(fname)


#
# == Class ImgsGreyscale ===============================================
class ImgsGreyscale:
    """These images are all greyscale with shape (z, y, x)"""

    def __init__(self) -> None:
        pass


#
# == Class ImgsColored =================================================
class ImgsColored:
    """These images are all colored with shape (z, y, x, 3)"""

    def __init__(self) -> None:
        pass


#
# == Class ImgsSameSize ================================================
class ImgsSameSize(ImgsImport):
    """These images all have the same Size with"""

    def __init__(
        self,
        path: str | Path = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(path, verbose)

        ### Check if all images have the same size
        self._check_size()

    def _check_size(self) -> None:
        """Check if all images have the same size"""
        sizes = [img.shape for img in self.imgs]
        if not all([size == sizes[0] for size in sizes]):
            raise ValueError(
                f"Not all images have the same size. Found these {set(sizes)}!"
            )
