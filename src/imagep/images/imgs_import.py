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
    from imagep.processing.pipeline import Pipeline
    from imagep.processing.preprocess import PreProcess


# %%
# == Class ImgsImport =====================================================
class ImgsImport:
    """Class for handling the most basic functionalities:
    - Imports of raw image data
    - Slicing of image stacks
    """

    PATH_PLACEHOLDER = "Source: Numpy array"
    IMGKEY_PLACEHOLDER = "NOKEY"

    def __init__(
        self,
        data: (
            str | Path | list[str | Path] | np.ndarray | list[np.ndarray] | Self
        ) = None,
        verbose: bool = True,
        ### KWS for importing from file
        **fileimport_kws,
    ) -> None:
        ### Make sure that either path or array is given
        self.verbose = verbose

        ### Allow path as an argument instead of data
        if "path" in fileimport_kws:
            data = fileimport_kws["path"]

        ### Init attributes, they will be set by the import functions
        self.path: str | Path | list[str | Path] = self.PATH_PLACEHOLDER
        self.imgs: np.ndarray = None
        self.imgkeys: list[str] = self.IMGKEY_PLACEHOLDER

        ### Configure import from path
        _importconfig = rc.RC_IMPORT
        _importconfig.update(fileimport_kws)
        self._fileimport_kws = _importconfig
        # > Remember the target dtype, could be useful
        self._dtype = self._fileimport_kws["dtype"]

        ### IMPORT data and convert it into dtype
        self._source_type = type(data)  # > Remember the source type
        self._import(data, dtype=self._dtype)

        ### Slicing
        # > Remember if this object has been sliced
        self._slice: bool | str = False
        self._shape_original: int = self.imgs.shape
        self._slice_indices: list[int] = list(range(self._shape_original[0]))

    #
    # == Import Source Data ============================================

    def _check_data_type(
        self,
        data: (
            str | Path | list[str | Path] | np.ndarray | list[np.ndarray] | Self
        ),
    ) -> None:
        """Check if data is a valid image source"""
        types = (str, Path, list, np.ndarray)
        m = (
            " Either (list of) path, (list of) array or an instance of Imgs"
            + " must be given."
        )
        if data is None:
            raise ValueError("No image source passed." + m)
        # > Z = PreProcess(data=Imgs)
        # > issubclass(data=PreProcess, self=Imgs)
        elif not (
            isinstance(data, types) or issubclass(type(self), type(data))
        ):
            raise ValueError(f"Unknown type of image source: {type(data)}." + m)

    def _import(
        self,
        data: (
            str | Path | list[str | Path] | np.ndarray | list[np.ndarray] | Self
        ),
        dtype: np.dtype,
    ) -> None:
        """Main import function. Calls the appropriate import function."""

        ### Check if data is a valid image source
        self._check_data_type(data)

        ### Import
        if isinstance(data, (str, Path)):
            self.imgkeys, self.imgs = self.from_path(data)
            self.path = data
        elif isinstance(data[0], (str, Path)):
            self.imgkeys, self.imgs = self.from_paths(data)
            self.path = data
        elif isinstance(data, (np.ndarray, list)) or isinstance(
            data[0], np.ndarray
        ):
            self.imgs = self.from_array(data)
            self.path = self.PATH_PLACEHOLDER
        ### Importing from Instance
        # > Z = PreProcess(data=Imgs)
        # > issubclass(self=PreProcess, data=Imgs)
        elif issubclass(type(self), type(data)):
            # !! Can't use data.verbose, because it's not set yet
            self.from_instance(data, verbose=self.verbose)

        ### dtype Conversion
        self.imgs = self.imgs.astype(dtype)

    #
    # == From Path, Array or Instance ==================================
    def from_paths(
        self, paths: list[str | Path]
    ) -> tuple[list[str], np.ndarray]:
        """Import images from a list of folders"""

        ### Convert if string
        paths = [Path(path) for path in paths]

        ### Check if paths are valid
        for i, path in enumerate(paths):
            if not path.exists():
                raise FileNotFoundError(f"Folder does not exist: {i}. '{path}'")

        ### Message
        if self.verbose:
            shortpaths = [self._shorten(path) for path in paths]
            print("=> Importing images from multiple folders:")
            for i, spath in enumerate(shortpaths):
                print(f"͘    {i}: '{spath}'")

        ### Import
        fks, _imgs = importtools.import_imgs_from_paths(
            paths, **self._fileimport_kws
        )
        # > Done
        if self.verbose:
            self._done_import_message(_imgs)

        ### Return
        return fks, _imgs

    def from_path(self, path: str | Path) -> tuple[str, np.ndarray]:
        """Import images from a folder"""

        ### Convert if string
        path = Path(path)

        ### Check if path is valid
        if not path.exists():
            raise FileNotFoundError(f"Folder does not exist: '{path}'")

        ### Message
        if self.verbose:
            print(f"=> Importing images from '{self._shorten(path)}' ...")

        ### Import
        # fks, _imgs = self._import_imgs_from_path(path, **self._fileimport_kws)
        fks, _imgs = importtools.import_imgs_from_path(
            path, **self._fileimport_kws
        )

        # > Done
        if self.verbose:
            self._done_import_message(_imgs)

        ### Return
        return fks, _imgs

    @staticmethod
    def _done_import_message(imgs: np.ndarray) -> None:
        print(
            f"   Import DONE ({imgs.shape[0]} images,"
            f" {imgs.shape[1]}x{imgs.shape[2]},"
            f" {imgs.dtype})"
        )
        print()

    def from_array(self, array: np.ndarray | list) -> np.ndarray:
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
        return array

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

    @staticmethod
    def _shorten(path: str | Path) -> str:
        path = Path(path)
        return path.parent.name + "/" + path.name

    @property
    def path_short(self) -> str | list[str]:
        """Shortened path"""
        if self.path == self.PATH_PLACEHOLDER:
            return self.PATH_PLACEHOLDER
        elif isinstance(self.path, (str | Path)):
            return self._shorten(self.path)
        elif isinstance(self.path, list):
            return [self._shorten(path) for path in self.path]

    #
    # == Import From Files =============================================

    @staticmethod
    def _import_imgs_from_path(
        path: str | Path, **fileimport_kws
    ) -> tuple[list[str], np.ndarray]:
        """Import z-stack from a folder"""
        return importtools.import_imgs_from_path(path, **fileimport_kws)

    # @staticmethod
    # def _import_imgs_from_path(path: Path, dtype: np.dtype) -> np.ndarray:
    #     """Import z-stack from a folder"""

    #     ### Get all txt files
    #     txts = list(path.glob("*.txt"))

    #     ### sort txts by number
    #     txts = sorted(txts, key=lambda x: int(x.stem.split("_")[-1]))

    #     ### Invert, since the first image is the bottom one
    #     txts = txts[::-1]

    #     ### Import all txt files
    #     imgs = []
    #     for txt in txts:
    #         imgs.append(importtools.from_txt(txt))

    #     ### Convert to numpy array
    #     imgs = np.array(imgs, dtype=dtype)

    #     return imgs

    #
    # == Access/Slice Images ===========================================

    # @property
    # def stack_raw(self) -> np.ndarray:
    #     return self._import_imgs_from_path()[1]

    def __iter__(self):
        return iter(self.imgs)

    def __getitem__(self, val: slice) -> Self | "Imgs" | "PreProcess":
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
            indices = range(*val.indices(self._shape_original[0]))
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
