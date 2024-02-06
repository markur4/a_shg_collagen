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
def import_imgs_from_paths(paths: list[str | Path], **import_kws) -> np.ndarray:
    """Imports from multiple paths by appending stacks from different
    folders onto another, and also returning their keys from filename
    data
    """
    ### Image stacks from multiple folders
    # > fks = filekeys
    # > fk = filekey
    imgs_nested: list[np.ndarray] = []
    fks_nested: list[list[str]] = []
    for path in paths:
        fks, _imgs = import_imgs_from_path(path=path, **import_kws)
        imgs_nested.append(_imgs)
        fks_nested.append(fks)
    _imgs = [import_imgs_from_path(path, **import_kws)[1] for path in paths]

    flatten = lambda x: [item for row in x for item in row]

    ### Flatten filekeys and imgs
    fks = flatten(fks_nested)
    _imgs = np.array(flatten(imgs_nested))
    
    return 

    # filekeys_flat = [filekeys[i] for i in range(len())]

    ###


def import_imgs_from_path(
    path: Path,
    fname_pattern: str = "",
    fname_extension: str = "",
    sort: bool = True,
    sortkey: Callable = None,
    invertorder: bool = True,
    dtype: np.dtype = rc.DTYPE_DEFAULT,
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
    (
        fname_extension
        if fname_extension.startswith(".")
        else "." + fname_extension
    )

    ### Define filepattern
    pattern = fname_pattern if fname_pattern else "*" + fname_extension

    ### Get all files
    filepaths = list(path.glob(pattern))

    ### sort txts by number
    if sort:
        filepaths = sorted(filepaths, key=sortkey)

    ### Invert if the first image is the bottom one
    if invertorder:
        filepaths = filepaths[::-1]

    ### Pick the right function to import
    import_func = function_from_format(fname_extension)

    ### Import all files
    _imgs = [
        import_func(path, dtype=dtype, **importfunc_kws) for path in filepaths
    ]
    _imgs = np.array(_imgs)  # > list to array

    ### Get the keys to identify individual images
    filekeys = ["" for _ in filepaths]  # > Initialize

    if not sortkey is None:
        # > Get the parents of the path
        folderkey = str(path.parent.name + "/" + path.name)
        filekeys = [f"{folderkey}: {sortkey(path)}" for path in filepaths]

    return filekeys, _imgs


def function_from_format(fname_extension: str) -> Callable:
    """Pick the right function to import the fname_extension"""

    if fname_extension == ".txt":
        return txtfile_to_array
    if fname_extension in (".tif"):
        return imgfile_to_array
    else:
        raise ValueError(f"fname_extension {fname_extension} not supported.")


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
