#
# %%
# import os
from typing import Self

# from pprint import pprint

from collections import OrderedDict

# > Processpool for CPU-bound tasks, ThreadPool for IO-bound tasks
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import skimage as ski


# > Local
import imagep._rc as rc

# from imagep.images.imgs import Imgs
import imagep._utils.utils as ut

# from imagep._utils.subcache import SubCache
# from imagep.processing.transforms import Transform
from imagep._plots.imageplots import imshow
from imagep.processing.pipeline import Pipeline
from imagep.processing.background import Background


# # %%
# # == Cache ===========================================================

# ### Location
# location = os.path.join(os.path.expanduser("~"), ".cache")

# ### Subcache
# CACHE_PREPROCESS = SubCache(
#     location=location,
#     subcache_dir="preprocess",
#     verbose=True,
#     compress=9,
#     bytes_limit="3G",  # > 3GB of cache, keeps only the most recent files
# )


# %%
# == CLASS PREPROCESSING ===============================================
class PreProcess(Pipeline):
    def __init__(
        self,
        *imgs_args,
        denoise=False,
        normalize=True,
        subtract_bg: bool = False,
        subtract_bg_kws: dict = dict(method="triangle", sigma=1.5),
        remove_empty_slices: bool = True,
        ### Imgs kws
        **imgs_process_kws,
    ) -> None:
        """Preprocessing pipeline for image stacks

        :param normalize: Wether to normalize values between 1 and 0,
            defaults to True
        :type normalize: bool, optional

        :param sample_index: Index of sample image to use for tracking
            the processing steps, defaults to 6
        :type sample_index: int, optional
        """
        super().__init__(*imgs_args, **imgs_process_kws)

        ### Check Arguments
        ut.check_arguments(subtract_bg_kws, ["method", "sigma"])

        ### Collect all preprocessing kws
        self.kws_preprocess = {
            "denoise": denoise,
            "normalize": normalize,
            "subtract_bg": subtract_bg,
            "subtract_bg_kws": subtract_bg_kws,
            "remove_empty_slices": remove_empty_slices,
        }

        ### Document History of processing steps
        self.history: OrderedDict = self._init_history(**self.kws_preprocess)
        # > List of preprocessing steps
        self._pp_steps = list(self.history.keys())[1:]
        
        ### Access to background
        self._background = Background(imgs=self.imgs, verbose=self.verbose)
        
        ### Execute !
        # self.imgs = self.preprocess(cache_preprocessing=cache_preprocessing)
        self.preprocess()

    # == Access to background functions and methods ====================
    @property
    def background(self):
        ### Keep imgs up-to-date
        self._background.imgs = self.imgs
        self._background.verbose = self.verbose
        ### Preserve background instance
        return self._background

    #
    # == Preprocess MAIN ===============================================

    def preprocess(self) -> np.ndarray:

        if self.verbose:
            print("=> Pre-processing ...")

        ### Shorten kws
        kws = self.kws_preprocess

        ### Get Sample of image before preprocessing
        self.capture_snapshot("before preprocessing")

        ### Denoise
        if kws["denoise"]:
            self.imgs = self.filter.denoise()
            self.capture_snapshot("Denoise")

        ### Subtract Background
        if kws["subtract_bg"]:
            self.imgs = self.background.subtract_threshold(
                **kws["subtract_bg_kws"]
            )
            self.capture_snapshot("subtract_bg")

        ### Normalize
        if kws["normalize"]:
            self.imgs = self.imgs / self.imgs.max()
            self.capture_snapshot("normalize")

        ### Remove empty slices
        if kws["remove_empty_slices"]:
            self.imgs = self.remove_empty_slices(imgs=self.imgs)


    #
    # === HISTORY ====================================================

    def _init_history(self, **preprocess_kws) -> OrderedDict:
        """Update the history of processing steps"""

        OD = OrderedDict()

        OD["Import"] = f"'{self.path}'"

        if preprocess_kws["denoise"]:
            OD["Denoising"] = "Non-local means"

        if preprocess_kws["subtract_bg"]:
            kws = preprocess_kws.get("subtract_bg_kws", dict())
            OD["BG Subtraction"] = (
                f"Calculated threshold (method = {kws['method']}) of"
                f" blurred images (gaussian filter, sigma = {kws['sigma']})."
                " Subtracted threshold from images and set negative"
                " values to 0"
            )

        if preprocess_kws["normalize"]:
            OD["Normalization"] = (
                "Division by max value of every image in folder"
            )

        if preprocess_kws["remove_empty_slices"]:
            OD["Empty Removal"] = (
                "Removed empty slices from stack. An entropy filter was"
                " applied to images. Images were removed if the 99th"
                " percentile of entropy was lower than"
                " than 10% of max entropy found in all images"
            )

        return OD

    def _history_to_str(self) -> str:
        """Returns the history of processing steps"""

        string = ""
        for i, (k, v) in enumerate(self.history.items()):
            string += f"  {i+1}. {k}: ".ljust(23)
            string += v + "\n"
        return string

    #
    # == __repr__ ======================================================

    @staticmethod
    def _info_brightness(imgs: np.ndarray) -> list:
        """Returns info about brightness for a Stack"""
        ### Formatting functions
        just = lambda s: ut.justify_str(s, justify=ut._JUSTIFY)
        form = lambda num: ut.format_num(num, exponent=ut._EXPONENT)

        ### Ignore background (0)
        # > or it'll skew statistics when bg is subtracted
        imgs_bg = imgs[imgs > 0.0]

        return [
            just("min (BG)") + f"{form(imgs.min())}",
            just("min, max (Signal)")
            + f"{form(imgs_bg.min())}, {form(imgs_bg.max())}",
            just("mean ± std")
            + f"{form(imgs_bg.mean())} ± {form(imgs_bg.std())}",
            just("median [IQR]")
            + f"{form(np.median(imgs_bg))} [{form(np.quantile(imgs_bg, .25))}"
            + f"- {form(np.quantile(imgs_bg, .75))}]",
        ]

    @property
    def _info(self) -> str:
        """String representation of the object for __repr__"""
        ### Formatting functions
        just = lambda s: ut.justify_str(s, justify=ut._JUSTIFY)
        form = lambda num: ut.format_num(num, exponent=ut._EXPONENT)

        imgs = self.imgs

        ### Check if background was subtracted
        bg_subtracted = str(self.kws_preprocess["subtract_bg"])

        ### Fill info
        ID = OrderedDict()

        ID["Data"] = [
            "=== Data ===",
            just("folder") + str(self.path_short),
            just("dtype") + str(imgs.dtype),
            just("shape") + str(imgs.shape),
            just("images") + str(len(imgs)),
        ]
        ID["Size"] = [
            "=== Size [µm] ===",
            just("pixel size xy") + f"{form(self.pixel_size)}",
            just("width, height (x,y)")
            + f"{form(self.x_µm)}, {form(self.y_µm)}",
        ]
        ID["Brightness"] = [
            "=== Brightness ===",
            just("BG subtracted") + bg_subtracted,
        ] + self._info_brightness(self.imgs)

        ID["History"] = [
            "=== Processing History ===",
            self._history_to_str(),
        ]

        return ID

    @staticmethod
    def _info_to_str(info: dict | OrderedDict) -> str:
        ### join individual lines
        string = ""
        for v in info.values():
            string += "\n".join(v) + "\n\n"

        return string

    @property
    def info(self) -> None:
        """Return string representation of the object"""
        print(self._info_to_str(self._info))

    def __str__(self) -> str:
        return self._info_to_str(self._info)

    def __repr__(self) -> str:
        """Remove memory address and class name so joblib doesn't redo
        the preprocessing
        """
        return f'< Images from "{self.path_short}">'

    #
    # == Z-Stack optimizations ============================================
    def remove_empty_slices(
        self, imgs: np.ndarray = None, threshold=0.10
    ) -> np.ndarray:
        """Empty images are those with a standard deviation lower than
        threshold (1%) of max standard deviation"""

        if self.verbose:
            print("=> Removing empty slices ...")

        ### Use self.imgs if imgs is None
        imgs = self.imgs if imgs is None else imgs

        ### Perform entropy filtering
        _imgs = self.filter.entropy(imgs=imgs)

        ### Get 99th percentiles for each image
        percentiles = np.percentile(_imgs, 99, axis=(1, 2))

        ### Show
        if rc.DEBUG:
            self.imshow(imgs=imgs, slice=3)  # original
            self.imshow(imgs=_imgs, slice=3)  # filtered
            plt.plot(percentiles / percentiles.max(), label="99percentiles")
            plt.axhline(threshold, color="red", label="threshold")
            plt.legend()

        ### Take only those slices where 99th percentile is above threshold
        imgs = imgs[percentiles > threshold]

        return imgs


#
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
    I = 17
    Z = PreProcess(
        data=path,
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(
            method="triangle",
            sigma=1.5,
            bg_per_img=True,
        ),
        scalebar_microns=50,
        snapshot_index=I,
        **kws,
    )
    print(Z.imgs.shape)
    print(Z._shape_original)

    # plt.imshow(zstack.stack[0])

    # %%
    ### Check history
    Z.info
    # %%
    Z.history

    # %%
    ### Check snapshots
    # Z.snapshots_array
    # %%
    Z.plot_snapshots()
    
    #%%
    # print(Z.background.threshold)

    
    # %%
    # imshow(Z.snapshots_array)
    # %%
    # imshow(Z.filter.denoise()[I])

    # %%
    # Z[I].imshow()

    # imshow(Z.snapshots_array)
    # %%
    hä

    # %%
    Z.kws_preprocess

    # %%
    ### Check __repr__
    Z._info

    # %%
    Z_bg = PreProcess(path, subtract_bg=True, **kws)

    # %%
    Z.info

    # %%
    Z_bg

    # %%
    print(Z)

    # %%
    # HÄÄÄÄ

    # %%
    Z.mip(axis=0, show=True)  # ' z-axis
    Z.mip(axis=1, show=True)  # ' x-axis
    Z.mip(axis=2, show=True)  # ' y-axis
    # %%
    print(Z.x_µm)
    print(Z.y_µm)
    # print(Z.z_µm)
    print(Z.pixel_size)
    print(Z.spacing)

    # %%
    #:: Denoising makes background subtraction better
    Z_d = PreProcess(
        path,
        denoise=True,
        **kws,
    )
    # Z_d_bg = PreProcess(
    #     path,
    #     denoise=True,
    #     background_subtract=0.06,  # > In percent of max brightness
    #     **kws,
    # )
    # %%
    Z_d.info

    # %%
    Z_d.mip()

    # %%
    #:: what's better to flatten background: denoise or blurring?

    S = Z_d.blur(sigma=1)
    plt.imshow(S[7])

    # %%
    histkws = dict(bins=200, log=False, alpha=0.4)

    plt.hist(Z.imgs.flatten(), label="raw", **histkws)
    plt.hist(S.flatten(), label="blur", **histkws)
    plt.legend()

    # %%
    plt.hist(Z.imgs.flatten(), label="raw", **histkws)
    plt.hist(Z_d.imgs.flatten(), label="denoise", **histkws)
    plt.legend()

    # %%
    plt.hist(S.flatten(), label="blur", **histkws)
    plt.hist(Z_d.imgs.flatten(), label="denoise", **histkws)
    plt.legend()

    # Z.brightness_distribution()
    # Z_d.brightness_distribution()
    # Z_d_bg.brightness_distribution()

    # %%
    print(ski.filters.threshold_triangle(Z.imgs))
    print(ski.filters.threshold_triangle(S))
    print(ski.filters.threshold_triangle(Z_d.imgs))

    # %%
    #:: Denoising preserves textures!
    # mip = Z.mip(ret=True)
    # mip_d = Z_d.mip(ret=True)
    # mip_d_bg = Z_d_bg.mip(ret=True)

    # %%
    # sns.boxplot(
    #     [
    #         mip.flatten(),
    #         # mip_d.flatten(),
    #         mip_d_bg.flatten(),
    #     ],
    #     showfliers=False,
    # )
