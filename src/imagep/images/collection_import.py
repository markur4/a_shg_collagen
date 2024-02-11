"""Classes to handle raw image formats and slicing"""

# %%
from typing import Self, TYPE_CHECKING

import copy

from pathlib import Path
import numpy as np

# > Local
import imagep._utils.utils as ut
import imagep._rc as rc
import imagep.images.importtools as importtools
from imagep.images.mdarray import mdarray

if TYPE_CHECKING:
    import imagep as ip
    from imagep.images.collection import Collection

    # from imagep.processing.pipeline import Pipeline
    from imagep.processing.preprocess import PreProcess
    from imagep.images.list_of_arrays import ListOfArrays


# %%
# == Class CollectionImport =====================================================
class CollectionImport:
    """Class for handling the most basic functionalities:
    - Imports of raw image data
    - Slicing of image stacks
    """

    FOLDER_PLACEHOLDER = ["Source: Numpy array"]
    IMGKEY_PLACEHOLDER = {"Source: Numpy array": "unnamed"}

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
        self.folder: str | Path | list[str | Path] = self.FOLDER_PLACEHOLDER
        self.imgs: ip.mdarray | ListOfArrays = None
        self.imgnames: dict[str, str] = self.IMGKEY_PLACEHOLDER

        ### Configure import from path
        _importconfig = copy.deepcopy(rc.IMPORTCONFIG)
        _importconfig.update(fileimport_kws)
        self._fileimport_kws = _importconfig

        ### IMPORT data and convert it into dtype
        self._import(data)

        ### Slicing
        # > Remember if this object has been sliced
        self._slice: bool | str = False
        self._shape_original: tuple[int | set] = self.imgs.shape
        self._slice_indices: list[int] = list(range(self._shape_original[0]))

    #
    # == ImgsDict ======================================================
    @property
    def imgs_dict(self) -> dict[str, np.ndarray]:
        """Returns dictionary with short folder names as keys and the
        images as values by retrieving metadata from the images"""

        D = {shortpath: [] for shortpath in self.path_short}
        for img in self.imgs:
            # folder = img.folder
            D[img.folder].append(img)
        return D

    #
    # == Path ==========================================================

    @property
    def path_short(self) -> list[str]:
        """Shortened path"""
        if self.folder == self.FOLDER_PLACEHOLDER:
            return self.FOLDER_PLACEHOLDER
        elif isinstance(self.folder, (str | Path)):
            return [ut.shortenpath(self.folder)]
        elif isinstance(self.folder, list):
            return [ut.shortenpath(path) for path in self.folder]

    @property
    def imgname_dict(self) -> dict[str, str]:
        """Returns a dictionary with short folder names as keys and the
        list of image names as values"""
        if self.imgnames == self.IMGKEY_PLACEHOLDER:
            return {self.IMGKEY_PLACEHOLDER: self.IMGKEY_PLACEHOLDER}
        elif isinstance(self.imgnames, list):
            return {self.path_short[0]: self.imgnames}

    #
    # == Import Data ===================================================

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
    ) -> None:
        """Main import function. Calls the appropriate import function."""

        ### Check if data is a valid image source
        self._check_data_type(data)

        ### Import
        if isinstance(data, (str, Path)):
            data = [data]
        if isinstance(data[0], (str, Path)):
            self.imgnames, self.imgs = self.from_folders(data)
            self.folder = data
        elif isinstance(data, (np.ndarray, list)) or isinstance(
            data[0], np.ndarray
        ):
            self.imgnames = self.IMGKEY_PLACEHOLDER
            self.imgs = self.from_array(data)
            self.folder = self.FOLDER_PLACEHOLDER
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
                + f" ({data.imgs.shape[0]} images "
                + f" {data.imgs.shape[1]}x{data.imgs.shape[2]},"
                + f" {data.imgs.dtype}; "
                + f" from: '{data.path_short}') ..."
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

    def __getitem__(
        self, val: int | slice | tuple | list
    ) -> Self | "Collection" | "PreProcess":
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
        elif isinstance(val, (list, tuple)):
            _self.imgs = self.imgs[list(val), ...]
            indices = val
        # > or Z[[1,2,5]] pick multiple images
        # elif isinstance(val, list):
        #     _self.imgs = self.imgs[val, ...]
        #     indices = val

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
    # !! End class Imgs ================================================


# %%
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
