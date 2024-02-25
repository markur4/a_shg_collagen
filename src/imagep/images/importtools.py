""" Utility functions to import images from various formats. """

# %%
from typing import Callable

import re

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

import skimage.io as sk_io

# import skimage as ski


# > Local
import imagep.types as T
import imagep._utils.utils as ut
import imagep._configs.rc as rc
from imagep._plots.imageplots import imshow
from imagep.arrays.mdarray import mdarray
from imagep.arrays.l2Darrays import l2Darrays


#
# == Multiple Images, Multiple Folders =================================
def arrays_from_folderlist(
    folders: list[str | Path],
    imgname_position: int | list[int] = 0,
    **import_kws,
) -> tuple[list[str], np.ndarray]:
    """Imports from multiple paths by appending stacks from different
    folders onto another, and also returning their keys from filename
    data
    """
    ### If just one imagekey position, make a list
    if isinstance(imgname_position, int):
        imgname_position = [imgname_position for _ in range(len(folders))]

    ### raise error if imgkey_positions and folders have different lengths
    # > and return a list of duplicated integers with same length
    imgname_position = ut.check_samelength_or_number(
        key="imgkey_positions",
        val=imgname_position,
        target_key="folders",
        target_n=len(folders),
    )

    imgs_dict: dict[str, np.ndarray] = {}
    imgnames_dict: dict[list[str]] = {}
    for path, imgkey_position in zip(folders, imgname_position):
        _imgkeys, _imgs = arrays_from_folder(
            folder=path,
            imgname_position=imgkey_position,
            **import_kws,
        )
        p = ut.shortenpath(path)
        imgs_dict[p] = _imgs
        imgnames_dict[p] = _imgkeys

    ### Flatten filekeys and imgs
    flatten = lambda x: [item for row in x for item in row]
    imgs = flatten(list(imgs_dict.values()))

    ### Convert to ListOfArrays
    imgs = l2Darrays(arrays=imgs)

    ### Assign index position to each image
    imgs = _assign_index_to_imgs(imgs)

    #!! don't use imgdict in parallel to imgs, to prevent confusion,
    #!! construct a new dict from imgs
    return imgnames_dict, imgs


def _assign_index_to_imgs(imgs: l2Darrays) -> l2Darrays:
    """Assigns index position to each image"""
    max_index = len(imgs) - 1
    for i, img in enumerate(imgs):
        img.index = (i, max_index)
    return imgs


# %%
# == Multiple Images, One Folder =======================================
def arrays_from_folder(
    folder: Path,
    fname_pattern: str = "",
    fname_extension: str = "",
    sort: bool = True,
    imgname_position: int = 0,
    invertorder: bool = True,
    dtype: np.dtype = rc.DTYPE,
    **importfunc_kws,
) -> np.ndarray:
    """Import z-stack from a folder"""

    ### Make sure either fname_extension or fname_pattern is given
    if not fname_extension and not fname_pattern:
        raise ValueError(
            "Either arguments must be given:"
            " 'fname_pattern' or 'fname_extension'."
        )

    ### Make sure fname_extension starts with a dot
    if not fname_extension.startswith("."):
        fname_extension = "." + fname_extension

    ### Define filepattern
    pattern = fname_pattern if fname_pattern else "*" + fname_extension

    ### Get all files and sort them
    _imgpaths = list(folder.glob(pattern))
    if len(_imgpaths) == 0:
        raise ValueError(f"No files found with pattern '{pattern}'")
    
    _imgpaths = _order_imgpaths(
        _imgpaths,
        sort=sort,
        imgname_position=imgname_position,
        invertorder=invertorder,
    )

    ### Pick the right function to import
    import_func = _pick_func_from_extension(fname_extension)

    ### Import all files
    _imgs = [
        import_func(path, dtype=dtype, **importfunc_kws) for path in _imgpaths
    ]
    _imgs = np.array(_imgs)  # > list to array

    ### Get the keys to identify individual images
    imgnames = [path.stem for path in _imgpaths]  # > Initialize
    if not imgname_position is None:
        # > Use a shorter name, if the position is given
        imgnames = _imgnames_from_imgpaths(_imgpaths, imgname_position)
    if not fname_pattern is None:
        imgnames = [str(name) + fname_pattern for name in imgnames]

    ### Add image names as metadata
    _imgs = [
        mdarray(
            array=img,
            name=imgname,
            folder=ut.shortenpath(folder),
        )
        for img, imgname in zip(_imgs, imgnames)
    ]

    return imgnames, _imgs


def _order_imgpaths(
    imgpaths: list[Path],
    sort: bool,
    imgname_position: int,
    invertorder: bool,
) -> list[Path]:
    """Function to re-order the image paths"""

    # > sort txts by number
    if sort:
        imgpaths = sorted(imgpaths, key=_get_sortkey(imgname_position))

    ### Invert if the first image is the bottom one
    if invertorder:
        imgpaths = imgpaths[::-1]

    return imgpaths


#
# == Extract Image names from Image paths ==============================


def _imgnames_from_imgpaths(
    imgpaths: list[Path],
    imgname_position: int,
) -> list[str]:
    """Get the keys to identify individual images"""

    return [_get_sortkey(imgname_position)(path) for path in imgpaths]


def _get_sortkey(imgkey_position: int = 0) -> Callable:
    return lambda path: _split_fname(path)[imgkey_position]


def _split_fname(path: str | Path) -> str:
    fname = Path(path).stem
    pattern = "|".join([" ", "_"])  # > Split at these characters
    split = re.split(pattern, fname)
    # > Convert to int if possible to sort by number
    split = [int(s) if s.isdigit() else s for s in split]
    return split


# %%
# == One path, One Image ===============================================
def array_from_path(
    path: str | Path,
    **importfunc_kws,
) -> np.ndarray:
    extension = _scan_extension(path)
    func = _pick_func_from_extension(extension)
    return func(path=path, **importfunc_kws)


def _scan_extension(path: str | Path) -> str:
    path = Path(path)
    return path.suffix


def _pick_func_from_extension(fname_extension: str) -> Callable:
    """Pick the right function to import the fname_extension"""

    if fname_extension == ".txt":
        return _array_from_txtfile
    if fname_extension in (".tif"):
        return _array_from_imgfile
    else:
        raise ValueError(f"fname_extension '{fname_extension}' not supported.")


# %%
# == Import from txt ===================================================
def _array_from_txtfile(
    path: str,
    skiprows: int = None,
    dtype: np.dtype = rc.DTYPE,
    **importfunc_kws,
) -> np.ndarray:
    """Import from a txt file."""

    kws = dict(dtype=dtype, **importfunc_kws)

    ### If skiprows is given, use it
    if not skiprows is None:
        return np.loadtxt(path, skiprows=skiprows, **kws)
    # > If not, try to skip up to 3 rows
    else:
        for i in range(4):  # > maximum 3 rows to skip
            try:
                return np.loadtxt(path, skiprows=i, **kws)
            except ValueError:
                continue
    ### If no skiprows work, raise an error
    raise ValueError(f"Could not import '{path}'.")


if __name__ == "__main__":
    import skimage as ski

    t = np.float32
    # path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_6.txt"
    # img = from_txt(path, type=t)
    # print(img.min(), img.max())
    # plt.imshow(img)
    # plt.show()

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_7.txt"
    img = _array_from_txtfile(path, dtype=t)
    print(img.min(), img.max())
    # plt.imshow(img)

    # %%
    ### Find smallest difference
    img_diff = ski.filters.sobel(img)
    print(img_diff.min(), img_diff.max())
    # plt.imshow(img_diff)


# %%
# == Import from Image formats =========================================
def _array_from_imgfile(
    path: str,
    dtype=rc.DTYPE,
    as_gray: bool = True,
) -> np.ndarray:
    """Import from image formats"""

    return sk_io.imread(path, as_gray=as_gray).astype(dtype)


if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/240201 Imunocyto/Exp. 1/Dmp1/D0 LTMC DAPI 40x.tif"
    img = _array_from_imgfile(path, dtype=np.float32)
    print(img.min(), img.max())
    # imshow(img)
    # imshow(img, cmap="gray")


# %%
# !! ==  tests =========================================================

if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough"
    imgkeys, imgs = arrays_from_folder(
        folder=Path(path),
        fname_extension=".txt",
        sort=True,
        imgname_position=1,
    )
    print(imgkeys)
    print(imgs[0].shape)
