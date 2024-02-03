""" Utility functions to import images from various formats. """

# %%
from typing import Callable

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

import skimage as ski


# > Local
from imagep._plots.imageplots import imshow
import imagep._utils.utils as ut
import imagep._rc as rc

# %%


def import_imgs_from_path(
    path: Path,
    fileformat: str,
    sort: bool = True,
    sortkey: Callable = None,
    invertorder: bool = True,
    dtype: np.dtype = rc.DTYPE_DEFAULT,
    **importfunc_kws,
) -> np.ndarray:
    """Import z-stack from a folder"""

    ### Make sure fileformat starts with a dot
    fileformat if fileformat.startswith(".") else "." + fileformat

    ### Get all files
    filepaths = list(path.glob("*" + fileformat))

    ### sort txts by number
    if sort:
        filepaths = sorted(filepaths, key=sortkey)

    ### Invert if the first image is the bottom one
    if invertorder:
        filepaths = filepaths[::-1]

    ### Pick the right function to import
    import_func = function_from_format(fileformat)

    ### Import all files
    _imgs = [
        import_func(path, dtype=dtype, **importfunc_kws) for path in filepaths
    ]
    _imgs = np.array(_imgs)  # > list to array

    return _imgs


def function_from_format(fileformat: str) -> Callable:
    """Pick the right function to import the fileformat"""

    if fileformat == ".txt":
        return txtfile_to_array
    if fileformat in (".tif"):
        return imgfile_to_array
    else:
        raise ValueError(f"Fileformat {fileformat} not supported.")


# %%
# == Import from txt ===================================================
def txtfile_to_array(
    path: str,
    skiprows: int = None,
    dtype: np.dtype = rc.DTYPE_DEFAULT,
) -> np.ndarray:
    """Import from a txt file."""

    if not skiprows is None:
        return np.loadtxt(path, skiprows=skiprows).astype(dtype)

    ### Skip rows until image is succesfully imported
    else:
        for i in range(3):  # > maximum 3 rows to skip
            try:
                return np.loadtxt(path, skiprows=i).astype(dtype)
            except:
                continue

    return np.loadtxt(path, skiprows=skiprows, dtype=dtype)


if __name__ == "__main__":
    t = np.float32
    # path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_6.txt"
    # img = from_txt(path, type=t)
    # print(img.min(), img.max())
    # plt.imshow(img)
    # plt.show()

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_7.txt"
    img = txtfile_to_array(path, dtype=t)
    print(img.min(), img.max())
    plt.imshow(img)

    # %%
    ### Find smallest difference
    img_diff = ski.filters.sobel(img)
    print(img_diff.min(), img_diff.max())
    plt.imshow(img_diff)

# %%
# == Import from Image formats =========================================


def imgfile_to_array(
    path: str,
    dtype=rc.DTYPE_DEFAULT,
    as_gray: bool = True,
) -> np.ndarray:
    """Import from image formats"""

    return ski.io.imread(path, as_gray=as_gray).astype(dtype)

if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/240201 Imunocyto/Exp. 1/Dmp1/D0 LTMC DAPI 40x.tif"
    img = imgfile_to_array(path, dtype=np.float32)
    print(img.min(), img.max())
    imshow(img)
    imshow(img, cmap="gray")