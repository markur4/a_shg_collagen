"""A class to handle background"""

# %%
from typing import TYPE_CHECKING, Tuple

import numpy as np
import skimage as ski

import matplotlib.pyplot as plt

# > Local
import imagep._configs.rc as rc
import imagep._utils.utils as ut

# from imagep.processing.pipeline import Pipeline
# from imagep.processing.preprocess import PreProcess
import imagep.processing.filters as filters
from imagep.images.l2Darrays import l2Darrays
import imagep.images.stack_meta as meta


# %%
# == Class: Background =================================================
class Background:

    def __init__(
        self,
        ### Images
        imgs: np.ndarray,
        verbose: bool = True,
        ### Settings
        method: str = "triangle",
        sigma: float = 1.5,
        per_img: bool = False,
    ) -> np.ndarray:
        # super().__init__(imgs=imgs, verbose=verbose) # !! no
        ### info
        self.imgs = imgs
        self.verbose = verbose

        ### Settings
        self.method = method
        self.sigma = sigma
        self.per_img = per_img

        # ### Check Arguments
        # ut.check_arguments(kws, ["method", "sigma", "per_img"])
        # self.method = kws.get("method", "triangle")  # > Method for thresholding
        # self.sigma = kws.get("sigma", 1.5)
        # self.per_img = kws.get("bg_per_img", False)

        ### Access to specific KWS

        ### Store information
        self.subtracted: bool = False  # > Was it subtracted..?
        self.threshold: list = [None]  # > Threshold(s) used for subtraction

    # == History =======================================================

    @property
    def _per_img_info(self) -> str:
        """Returns True if the background was subtracted per image"""
        if self.per_img:
            return "Individual thresholds per image"
        else:
            return "One threshold for the stack"
        # return self.kws.get("bg_per_img", False)

    @property
    def info(self) -> str:
        """Returns information about the background printable by
        _info"""

        form = lambda x: ut.format_num(x, exponent=rc.FORMAT_EXPONENT)

        if self.subtracted:
            ### Thresholds
            if self.per_img:
                mean = form(np.mean(self.threshold))
                std = form(np.std(self.threshold))
                threshold = f"{mean} ± {std}"
            else:
                threshold = f"{self.threshold[0]}"

            msg = f"{self._per_img_info} = {threshold}"
        elif self.subtracted is None:
            msg = "No background subtracted, since undefined"
        else:
            msg = "No background subtracted"

        return msg

    #
    # == Subtract MAIN functions =======================================
    
    # @meta.preserve_metadata() # !! No nested
    def subtract_threshold(
        self,
        method: str = None,
        sigma: float = None,
        per_img: bool = None,
    )-> np.ndarray:
        """Subtracts a threshold from the images"""

        ### Collect KWS
        kws = dict(
            method=self.method if method is None else method,
            sigma=self.sigma if sigma is None else sigma,
            per_img=self.per_img if per_img is None else per_img,
        )

        if self.verbose:
            msg = f"Subtracting threshold"
            msg_ap = f"(Method: {kws['method']}, sigma={kws['sigma']};"
            msg_ap += f" {self._per_img_info})"
            _imgs, threshold = ut._messaged_execution(
                f=_subtract_threshold,
                msg=msg,
                msg_after_points=msg_ap,
                filter_KWS=dict(imgs=self.imgs, **kws),
            )
        else:
            _imgs, threshold = _subtract_threshold(**kws)

        ### Store results
        self.subtracted = True
        self.threshold = threshold

        return _imgs

    def subtract_local_threshold(self, **kws):
        """Subtracts the local threshold from the images
        - rolling ball
        """
        raise NotImplementedError

    #
    # !! == End Class ==================================================


#
# == Static functions ==================================================


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

    ### Raise error if value is larger than the max value
    if value > img.max():
        raise ValueError(
            f"Background value {ut.format_num(value)} is larger than"
            f" the max value {ut.format_num(img.max())}. "
        )

    img_bg = img - value
    ### Set negative values to 0
    img_bg[img_bg < 0] = 0

    return img_bg

# @meta.preserve_metadata() #!! no nested metadata
def _subtract_threshold(
    imgs: np.ndarray,
    method: str,
    sigma: float,
    per_img=True,
) -> tuple[np.ndarray, list[float]]:
    """Subtracts a threshold (calculated by method) from the images"""
    kernel_3D = False if per_img else True

    ### Gaussian blur
    imgs_blurred = filters.blur(
        imgs, sigma=sigma, kernel_3D=kernel_3D, normalize=True
    )

    if not per_img:
        ### List with ONE threshold for the stack
        threshold = [get_threshold_from_array(imgs_blurred, method=method)]

        ### Subtract
        _imgs = [subtract(img, value=threshold) for img in imgs]

    else:
        ### Threshold for each image
        threshold = [
            get_threshold_from_array(img, method=method) for img in imgs_blurred
        ]
        ### Subtract
        _imgs = [subtract(img, value=t) for img, t in zip(imgs, threshold)]

    ### Convert to l2Darray
    # _imgs = np.array(_imgs, dtype=imgs.dtype)
    _imgs = l2Darrays(_imgs, dtype=imgs.dtype)
    return _imgs, threshold

    # !! == End Class ==================================================


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
        pixel_length=(1.5 * 115.4)/1024,
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
        fname_extension="txt",
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(
            method="triangle",
            sigma=1.5,
            per_img=True,
        ),
        remove_empty_slices=True,
        scalebar_length=50,
        snapshot_index=I,
        **kws,
    )
    print(Z.imgs.shape)
    print(Z._shape_original)
    
    #%%
    
    Z.imgs
    
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
