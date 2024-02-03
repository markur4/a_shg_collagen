"""A class to handle background"""

# %%
from typing import TYPE_CHECKING, Tuple

import numpy as np
import skimage as ski

# > Local
# from imagep.processing.preprocess import PreProcess
import imagep.processing.filters as filters


# %%
# == Class: Background =================================================
class Background:

    def __init__(self, imgs: np.ndarray, verbose: bool = True) -> np.ndarray:
        self.imgs = imgs
        self.verbose = verbose

        ### Store the threshold value
        self.threshold = None

    def subtract_threshold(self, method: str, sigma: float, bg_per_img=False):
        """Subtracts a threshold from the images"""
        _imgs, threshold = subtract_threshold(
            imgs=self.imgs,
            method=method,
            sigma=sigma,
            bg_per_img=bg_per_img,
        )
        self.threshold = threshold
        return _imgs

    def subtract_local_threshold(self, **kws):
        """Subtracts the local threshold from the images
        - rolling ball
        """
        raise NotImplementedError


#
# == BG Subtract ===================================================


def get_threshold_by_percentile(
    array: np.ndarray, percentile: int = 10
) -> np.float64:
    """Defines background as percentile of the stack"""
    return np.percentile(array, percentile, axis=0)


def get_threshold_by_percent(
    array: np.ndarray, percent: float = 0.05
) -> np.float64:
    """Defines a manual background as a percentage of the max value"""
    return array.max() * percent


def get_threshold_from_array(
    array: np.ndarray,
    method="triangle",
    **kws,
) -> np.float64:
    """Calculates a threshold for an array"""

    ### Apply Filters
    if method == "otsu":
        return ski.filters.threshold_otsu(array, **kws)
    elif method == "mean":
        return ski.filters.threshold_mean(array, **kws)
    elif method == "triangle":
        return ski.filters.threshold_triangle(array, **kws)
    elif method == "percentile":
        return get_threshold_by_percentile(array, **kws)
    elif method == "threshold":
        return get_threshold_by_percent(array, **kws)
    else:
        raise ValueError(f"Unknown method: {method}")


def subtract(img: np.ndarray, value: float) -> np.ndarray:
    """subtracts value from stack and sets negative values to 0"""
    img_bg = img - value
    ### Set negative values to 0
    img_bg[img_bg < 0] = 0

    return img_bg


def subtract_threshold(
    imgs: np.ndarray,
    # self,
    method: str,
    sigma: float,
    bg_per_img=False,
    # inplace=True,
) -> tuple[np.ndarray, float | list[float]]:
    """Subtracts a threshold (calculated by method) from the images"""
    kernel_3D = False if bg_per_img else True

    ### Gaussian blur
    imgs_blurred = filters.blur(
        imgs, sigma=sigma, kernel_3D=kernel_3D, normalize=True
    )

    if not bg_per_img:
        ### Threshold for the stack
        threshold: float = get_threshold_from_array(imgs_blurred, method=method)

        ### Subtract
        _imgs = [subtract(img, value=threshold) for img in imgs]

    else:
        ### Threshold for each image
        threshold: list = [
            get_threshold_from_array(img, method=method) for img in imgs_blurred
        ]
        ### Subtract
        _imgs = [subtract(img, value=t) for img, t in zip(imgs, threshold)]

    ### Convert to array
    _imgs = np.array(_imgs, dtype=imgs.dtype)
    return _imgs, threshold


# !! ===================================================================


# %%
# == TESTS ============================================================

if __name__ == "__main__":
    ### Import from a txt file.
    # > Rough
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    # > x_µm = fast axis amplitude * calibration =  1.5 V * 115.4 µm/V
    # > z_dist = n_imgs * stepsize = 10 * 0.250 µm
    kws = dict(
        # z_dist=10 * 0.250,
        x_µm=(1.5 * 115.4),
    )
    # > Detailed
    # path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    # kws = dict(
    #     # z_dist=2 * 0.250,  # > stepsize * 0.250 µm
    #     x_µm=1.5
    #     * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    # )
    # %%
    from imagep.processing.preprocess import PreProcess

    I = 4
    Z = PreProcess(
        data=path,
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(
            method="triangle",
            sigma=1.5,
            bg_per_img=True,
        ),
        remove_empty_slices=True,
        scalebar_microns=50,
        snapshot_index=I,
        **kws,
    )
    print(Z.imgs.shape)
    print(Z._shape_original)

    # plt.imshow(zstack.stack[0])

    # %%
    ### Check snapshots
    # Z.snapshots_array
    # %%
    Z.plot_snapshots()

    # %%
    print(Z.background.threshold)

    # %%
    import matplotlib.pyplot as plt

    plt.plot(Z.background.threshold)
