"""Classes to handle raw image formats and slicing"""

# %%
from typing import Self, TYPE_CHECKING

import copy

from pathlib import Path
import numpy as np

# > Local
import imagep._utils.utils as ut
import imagep._configs.rc as rc
import imagep.types as T
import imagep.images.importtools as importtools
from imagep.images.mdarray import mdarray
from imagep.images.l2Darrays import l2Darrays

if TYPE_CHECKING:
    import imagep as ip
    from imagep.images.stack import Stack

    # from imagep.processing.pipeline import Pipeline
    from imagep.processing.preprocess import PreProcess


# %%
# ======================================================================
# == Class StackImport =================================================
class StackImport:
    """Class for handling the most basic functionalities:
    - Imports of raw image data
    - Slicing of image stacks
    """

    FOLDER_PLACEHOLDER = ["Source: Numpy array"]
    IMGKEY_PLACEHOLDER = {"Source: Numpy array": "unnamed"}

    def __init__(
        self,
        data: T.source_of_imgs = None,
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
        self.folders: list[str | Path] = self.FOLDER_PLACEHOLDER
        self.imgs: ip.mdarray | l2Darrays = None
        self.imgnames: dict[str, str] = self.IMGKEY_PLACEHOLDER

        ### Configure import from path
        _importconfig = copy.deepcopy(rc.IMPORTCONFIG)
        _importconfig.update(fileimport_kws)
        self._fileimport_kws = _importconfig

        ### IMPORT data and convert it into dtype
        self._import(data)

        ### Slicing
        # > Remember if this object has been sliced
        self._sliced: bool | str = False
        self._shape_original: tuple[int | set] = self.imgs.shape
        self._slice_indices: list[int] = list(range(self._shape_original[0]))

    #
    # == ImgsDict ======================================================
    @property
    def imgs_dict(self) -> dict[str, np.ndarray]:
        """Returns dictionary with short folder names as keys and the
        images as values by retrieving metadata from the images"""

        D = {shortpath: [] for shortpath in self.paths_short}
        for img in self.imgs:
            # folder = img.folder
            D[img.folder].append(img)
        return D

    #
    # == Path ==========================================================

    @property
    def paths_short(self) -> list[str]:
        """Shortened path"""
        if self.folders == self.FOLDER_PLACEHOLDER:
            return self.FOLDER_PLACEHOLDER
        elif isinstance(self.folders, (str | Path)):
            return [ut.shortenpath(self.folders)]
        elif isinstance(self.folders, list):
            return [ut.shortenpath(path) for path in self.folders]

    @property
    def paths_pretty(self) -> str:
        return "\n".join(self.paths_short)

    @property
    def imgname_dict(self) -> dict[str, str]:
        """Returns a dictionary with short folder names as keys and the
        list of image names as values"""
        if self.imgnames == self.IMGKEY_PLACEHOLDER:
            return {self.IMGKEY_PLACEHOLDER: self.IMGKEY_PLACEHOLDER}
        elif isinstance(self.imgnames, list):
            return {self.paths_short[0]: self.imgnames}

    #
    # == Import Data ===================================================

    def _check_data_type(
        self,
        data: T.source_of_imgs,
    ) -> None:
        """Check if data is a valid image source"""
        types = (str, Path, list, T.array, l2Darrays)
        m = (
            " Either (list of) path, (list of) array or self"
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
    ) -> None:
        """Main import function. Calls the appropriate import function."""

        ### Check if data is a valid image source
        self._check_data_type(data)

        ### Import
        if isinstance(data, (str, Path)):
            data = [data]
        if isinstance(data[0], (str, Path)):
            self.imgnames, self.imgs = self.from_folders(data)
            self.folders = data
        elif isinstance(data, (np.ndarray, list)) or isinstance(
            data[0], np.ndarray
        ):
            self.imgnames = self.IMGKEY_PLACEHOLDER
            self.imgs = self.from_array(data)
            self.folders = self.FOLDER_PLACEHOLDER
        ### Importing from Instance
        # > Z = PreProcess(data=Imgs)
        # > issubclass(self=PreProcess, data=Imgs)
        elif issubclass(type(self), type(data)):
            # !! Can't use data.verbose, because it's not set yet
            self.from_instance(data, verbose=self.verbose)

    #
    # == From Path, Array or Instance ==================================
    def from_folders(
        self, folders: list[str | Path]
    ) -> tuple[list[str], np.ndarray | list[np.ndarray]]:
        """Import images from a list of folders"""

        ### Convert if string
        folders = [Path(path) for path in folders]

        ### Check if paths are valid
        for i, path in enumerate(folders):
            if not path.exists():
                raise FileNotFoundError(f"Folder does not exist: {i}. '{path}'")

        ### Message
        if self.verbose:
            shortpaths = [ut.shortenpath(path) for path in folders]
            print("=> Importing images from folder(s):")
            for i, spath in enumerate(shortpaths):
                print(f"   | {i}: '{spath}'")

        ### Import
        imgnames_dict, imgs = importtools.arrays_from_folderlist(
            folders, **self._fileimport_kws
        )

        # > Done
        if self.verbose:
            self._done_import_message(imgs)

        ### Return
        return imgnames_dict, imgs

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

        ### Convert if list
        waslist = False
        if isinstance(array, list):
            waslist = True
            array = np.array(array)

        ### Check for shape (z, y, x)
        if len(array.shape) != 3:
            raise ValueError(
                f"Array must have shape (z, y, x), not {array.shape}"
            )

        ### Convert to Mdarray
        array = mdarray(array)

        ### Message
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

    def from_instance(self, data: Self, verbose: bool) -> None:
        """Transfer images and path from another instance"""

        # !! This mustn't be self.verbose, because it's not set yet
        if verbose:
            print(
                f"=> Transferring attributes from an instance"
                + f" ({data.imgs.shapes[0]} images "
                + f" {data.imgs.shapes[1]}x{data.imgs.shapes[2]},"
                + f" {data.imgs.dtypes}; "
                + f" from: '{data.paths_short}') ..."
            )
        ### Full atrribute transfer
        for attr, value in data.__dict__.items():
            setattr(self, attr, value)

        if verbose:
            print("   Transfer DONE")
            print()

    #
    # == Import From Files =============================================

    @staticmethod
    def _import_imgs_from_path(
        path: str | Path, **fileimport_kws
    ) -> tuple[list[str], np.ndarray]:
        """Import z-stack from a folder"""
        return importtools.arrays_from_folder(path, **fileimport_kws)

    #
    # == Access/Slice Images ===========================================

    def __iter__(self):
        return iter(self.imgs)

    def __getitem__(self, val: T.indices) -> Self | "Stack" | "PreProcess":
        """Slice the z-stack and return a copied instance of this class.
        with a changes self.imgs.
        Slicing happens PRESERVING the dimension information. That means
        that the result is always a 3D array.
        """
        ### Create a copy of this instance
        _self = copy.deepcopy(self)

        ### Slice
        if isinstance(self.imgs, np.ndarray):  # If imgs is a numpy array
            # > Z[0]
            if isinstance(val, int):
                _self.imgs = self.imgs[[val], ...]
                indices = [val]
            # > Z[1:3]
            elif isinstance(val, slice):
                _self.imgs = self.imgs[val, ...]
                indices = range(*val.indices(self._shape_original[0]))
            # > Z[1,2,5] or Z[[1,2,5]]
            elif isinstance(val, (list, tuple)):
                _self.imgs = self.imgs[list(val), ...]
                indices = val
        elif isinstance(self.imgs, l2Darrays):
            # > Let the ListOfArrays handle the slicing
            # > Re-initialize as ListOfArrays to preserve the type
            imgs: T.array | l2Darrays = self.imgs[val]
            _self.imgs = l2Darrays(arrays=imgs)
            # > Z[0] or Z[1,2,5] or Z[[1,2,5]]
            if isinstance(val, int):
                indices = [val]
            elif isinstance(val, (list, tuple)):
                indices = val
            # > Z[1:3]
            elif isinstance(val, slice):
                indices = range(*val.indices(len(self.imgs)))
            # > Z[1,2,5]
        else:
            raise ValueError(f"Unknown type of imgs: {type(self.imgs)}")

        ### Remember how this object was sliced
        _self._sliced = str(val)
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
    # !! End class Imgs ================================================


# %%

#
# == Class ImgsColored =================================================
class StackColored(StackImport):
    """These images are all colored with shape (z, y, x, 3)"""

    def __init__(self, imgs) -> None:
        super().__init__(data=imgs)

        self.imgs_r = self.split_channel(0)
        self.imgs_g = self.split_channel(1)
        self.imgs_b = self.split_channel(2)

    def split_channel(self, channel: int) -> np.ndarray:
        """Split the images into the different color channels"""
        return self.imgs[..., channel]

    def merge_channels(self, imgs_r, imgs_g, imgs_b) -> np.ndarray:
        """Merge the images into the different color channels"""
        return np.stack([imgs_r, imgs_g, imgs_b], axis=-1)


#
# # == Class ImgsSameSize ================================================
# class ImgsSameSize(ImgsImport):
#     """These images all have the same Size with"""

#     def __init__(
#         self,
#         path: str | Path = None,
#         verbose: bool = True,
#     ) -> None:
#         super().__init__(path, verbose)

#         ### Check if all images have the same size
#         self._check_size()

#     def _check_size(self) -> None:
#         """Check if all images have the same size"""
#         sizes = [img.shape for img in self.imgs]
#         if not all([size == sizes[0] for size in sizes]):
#             raise ValueError(
#                 f"Not all images have the same size. Found these {set(sizes)}!"
#             )
