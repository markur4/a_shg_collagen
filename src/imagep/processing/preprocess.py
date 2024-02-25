#
# %%
# import os
from typing import Self

# from pprint import pprint

from collections import OrderedDict
from typing import TypedDict

# > Processpool for CPU-bound tasks, ThreadPool for IO-bound tasks
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import skimage as ski


# > Local
import imagep._configs.rc as rc
import imagep._utils.parameters as p

# from imagep.images.imgs import Imgs
import imagep._utils.utils as ut
import imagep.images.metadata as meta
import imagep.types as T
from imagep._plots.imageplots import imshow
from imagep.arrays.l2Darrays import l2Darrays

# from imagep._utils.subcache import SubCache
# from imagep.processing.transforms import Transform
from imagep.processing.process import Process
from imagep.processing.background import Background


# %%
# ### Config for preprocessing
# class PreProcessKWS(TypedDict):
#     denoise: bool
#     normalize: bool
#     subtract_bg: bool
#     subtract_bg_kws: dict
#     remove_empty_slices: bool


# if __name__ == "__main__":
#     config = PreProcessKWS(
#         denoise=True,
#         normalize=True,
#         subtract_bg=True,
#         subtract_bg_kws=dict(
#             method="triangle",
#             sigma=1.5,
#         ),
#         remove_empty_slices=True,
#     )
#     print(config.denoise)


# %%
# == CLASS PREPROCESSING ===============================================
class PreProcess(Process):
    def __init__(
        self,
        ### Imgs and Pipeline args
        *stack_args,
        ### Preprocessing kws
        median: bool = False,
        denoise: bool = True,
        normalize: bool | str = "img",
        subtract_bg: bool = False,
        subtract_bg_kws: dict = dict(
            method="otsu", sigma=3, per_img=False
        ),
        remove_empty_slices: bool = False,
        ### Imgs and pipeline kws
        **stack_kws,
    ) -> None:
        """Preprocessing pipeline for image stacks

        :param normalize: Wether to normalize values between 1 and 0,
            defaults to True
        :type normalize: bool, optional

        :param sample_index: Index of sample image to use for tracking
            the processing steps, defaults to 6
        :type sample_index: int, optional
        """
        super().__init__(*stack_args, **stack_kws)

        ### Access to background functionality
        self._background = Background(
            imgs=self.imgs, verbose=self.verbose, **subtract_bg_kws
        )

        ### Collect all preprocessing kws
        self.kws_preprocess = {
            "median": median,
            "denoise": denoise,
            "normalize": normalize,
            "subtract_bg": subtract_bg,
            # "subtract_bg_kws": subtract_bg_kws,
            "remove_empty_slices": remove_empty_slices,
        }

        ### Document History of processing steps
        self.history: OrderedDict = self._init_history(**self.kws_preprocess)
        # > List of preprocessing steps
        self.history_steps = list(self.history.keys())[1:]

        ### Execute !
        # self.imgs = self.preprocess(cache_preprocessing=cache_preprocessing)
        self.preprocess()

    # == Access to background functions and methods ====================
    @property
    def background(self):
        ### Keep imgs up-to-date
        self._background.imgs = self.imgs
        # self._background.kws = self.kws_preprocess["subtract_bg_kws"]
        self._background.verbose = self.verbose
        ### Preserve background instance
        return self._background

    #
    # == Preprocess MAIN ===============================================

    def preprocess(self) -> Self:

        if self.verbose:
            print(f"=> Pre-processing: {self.history_steps} ...")

        ### Shorten kws
        kws = self.kws_preprocess

        ### Get Sample of image before preprocessing
        self.capture_snapshot("before preprocessing")

        ### Median filter
        if kws["median"]:
            self.imgs = self.filter.median(kernel_radius=2, kernel_3D=False)
            self.capture_snapshot("median")

        ### Denoise
        if kws["denoise"]:
            self.imgs = self.filter.denoise()
            self.capture_snapshot("Denoise")

        ### Subtract Background
        if kws["subtract_bg"]:
            self.imgs = self.background.subtract_threshold()
            self.capture_snapshot("subtract_bg")

        ### Normalize
        if kws["normalize"]:
            self.imgs = self.normalize()
            self.capture_snapshot("normalize")

        ### Remove empty slices
        if kws["remove_empty_slices"]:
            self.imgs = self.remove_empty_slices()
        
        return self

    #
    # === HISTORY ====================================================

    def _init_history(self, **preprocess_kws) -> OrderedDict:
        """Update the history of processing steps"""

        od = OrderedDict()

        od["Import"] = f"'{self.folders}'"

        if preprocess_kws["median"]:
            od["Median Filter"] = (
                "Median filter using a 2D disk shaped kernel with radius of 2 pixels."
            )

        if preprocess_kws["denoise"]:
            od["Denoising"] = "Non-local means"

        if preprocess_kws["subtract_bg"]:
            # kws = self.background.kws
            od["BG Subtraction"] = (
                f"Calculated threshold (method = {self.background.method}) of"
                f" blurred images (gaussian filter, sigma = {self.background.sigma})."
                " Subtracted threshold from images and set negative"
                " values to 0"
            )

        if preprocess_kws["normalize"]:
            od["Normalization"] = (
                "Division by max value of every image in folder"
            )

        if preprocess_kws["remove_empty_slices"]:
            od["Remove Empty"] = (
                "Removed empty slices from stack. An entropy filter was"
                " applied to images. Images were removed if the 99th"
                " percentile of entropy was lower than"
                " than a threshold entropy found in all images."
                " The entropy threshold was calculated using Otsu's method."
            )

        return od

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
        just = lambda s: ut.justify_str(s, justify=rc.FORMAT_JUSTIFY)
        form = lambda num: ut.format_num(num, exponent=rc.FORMAT_EXPONENT)

        ### Ignore background (0)
        # > or it'll skew statistics when bg is subtracted
        # TODO: Change this after implementing list of 1D arrays
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
        just = lambda s: ut.justify_str(s, justify=rc.FORMAT_JUSTIFY)
        form = lambda num: ut.format_num(num, exponent=rc.FORMAT_EXPONENT)

        imgs = self.imgs

        # ### Check if background was subtracted
        # bg_subtracted = str(self.kws_preprocess["subtract_bg"])

        ### Fill info
        id = OrderedDict()

        id["Data"] = [
            "=== Data ===",
            just("folder") + str(self.paths_short),
            just("dtype") + str(imgs.dtype),
            just("shape") + str(imgs.shape),
            just("images") + str(len(imgs)),
        ]
        id["Size"] = [
            "=== Size [µm] ===",
            # just("pixel size xy") + f"{form(self.pixel_length)}",
            just("width, height (x,y)"),
            # + f"{form(self.x_µm)}, {form(self.y_µm)}",
        ]
        id["Brightness"] = [
            "=== Brightness ===",
            just("BG subtracted") + self.background.info,
        ] + self._info_brightness(self.imgs)

        id["History"] = [
            "=== Processing History ===",
            self._history_to_str(),
        ]

        return id

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
        return f'< Images from "{self.paths_short}">'

    #
    # == Normalize =====================================================
    # @meta.preserve_metadata()
    def normalize(self, across: str | bool = None) -> np.ndarray:
        ### Get arg
        # print(f"{self.imgs[0].metadata=}")

        ### Handle args
        across = self.kws_preprocess["normalize"] if across is None else across
        across = p.handle_param(p.ACROSS, param=across, funcname="normalize")

        ### Normalize each individual image
        if across == "img":
            return l2Darrays([img / img.max().item() for img in self.imgs])

        ### Normalize across the whole stack
        elif across == "stack":
            # imgs = l2Darrays(self.imgs / self.imgs.max().item())
            # print(f"{imgs[0].metadata=}")
            # return imgs
            return l2Darrays(self.imgs / self.imgs.max().item())
        else:
            raise ValueError(
                f"across = '{across}' not recognized. Use 'img' or 'stack'"
            )

    #
    # == Z-Stack optimizations ============================================
    def remove_empty_slices(self, imgs: T.array = None) -> np.ndarray:
        """Empty images are those with a standard deviation lower than
        threshold (1%) of max standard deviation"""
        # print(f"{type(self.imgs)=}")

        if self.verbose:
            print("=> Removing empty slices ...")

        ### Use self.imgs if imgs is None
        imgs = self.imgs if imgs is None else imgs

        ### Perform entropy filtering
        _imgs = self.filter.entropy(imgs=imgs, normalize=False)
        # print(f"{type(_imgs)=}")

        ### Get 99th percentiles for each image
        percentiles = np.percentile(_imgs, 99, axis=(1, 2))

        ### Define threshould through otsu in percentiles
        threshold = ski.filters.threshold_otsu(percentiles)

        ### Show
        if rc.DEBUG:
            self.imshow(imgs=imgs, slice=3)  # original
            self.imshow(imgs=_imgs, slice=3)  # filtered
            plt.plot(percentiles / percentiles.max(), label="99percentiles")
            plt.axhline(threshold, color="red", label="threshold")
            plt.legend()

        ### Take only those slices where 99th percentile is above threshold
        # print(f"before {type(imgs)=}", f"{type(imgs[0])=}")
        # print(imgs.shape)
        # print(percentiles > threshold)
        imgs = imgs[percentiles > threshold]

        # print(f"afta {type(imgs)=}", f"{type(imgs[0])=}")
        # print(imgs.shape)

        return imgs


#
# !! == End Class ==================================================


# %%
# == TESTS ============================================================

if __name__ == "__main__":
    import imagep as ip

    parent = (
        "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/"
    )

    ### Import from a txt file.
    # > Rough
    path = parent + "1 healthy z-stack rough/"
    # > x_µm = fast axis amplitude * calibration =  1.5 V * 115.4 µm/V
    # > z_dist = n_imgs * stepsize = 10 * 0.250 µm
    kws = dict(
        # z_dist=10 * 0.250,
        pixel_length=(1.5 * 115.4)
        / 1024,
    )
    # > Detailed
    # path = parent + "2 healthy z-stack detailed/"
    # kws = dict(
    #     # z_dist=2 * 0.250,  # > stepsize * 0.250 µm
    #     x_µm=1.5
    #     * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    # )

    # %%
    I = 8
    Z = PreProcess(
        data=path,
        fname_extension=".txt",
        imgname_position=1,
        denoise=True,
        normalize="stack",
        subtract_bg=True,
        subtract_bg_kws=dict(
            method="otsu",
            sigma=3,
            per_img=False,
        ),
        scalebar_length=10,
        snapshot_index=I,
        remove_empty_slices=True,
        **kws,
    )
    #%%
    # Z = Z.preprocess()
    # %%
    Z_RAW = PreProcess(
        data=path,
        fname_extension=".txt",
        imgname_position=1,
        denoise=False,
        normalize=False,
        subtract_bg=False,
        scalebar_length=10,
        snapshot_index=I,
        remove_empty_slices=False,
        **kws,
    )
    #%%
    # Z_RAW = Z_RAW.preprocess()
    # %%
    print(type(Z.imgs))
    print(type(Z.imgs[0]))

    # %%
    Z.plot_snapshots(save_as="1_preprocessing.pdf")

    # %%
    Z.scalebar_length

    # %%
    Z.plot_histogram(save_as="2_histogram.pdf")
    Z_RAW.plot_histogram(save_as="2_histogram_raw.pdf")
    # %%
    print(Z.imgs.shape)
    print(Z._shape_original)
    print(type(Z.imgs))
    print(type(Z.imgs[0]))
    # print(Z.info)

    # %%
    Z[I].imshow()
    # plt.imshow(zstack.stack[0])

    # %%
    ### Present entropy filter
    _entr = Z.filter.entropy(normalize=False)
    _entr_raw = Z_RAW.filter.entropy(normalize=False)
    # %%
    m = _entr.max()
    ip.imshow(
        [
            Z.imgs[I],
            _entr_raw[I] / m,
            _entr[I - 2] / m,
            Z.imgs[2 - 2],
            _entr_raw[2] / m,
            _entr[2 - 2] / m,
        ],
        save_as="3_entropy.pdf",
        max_cols=3,
    )

    # %%
    percentiles99 = np.percentile(_entr_raw, 99, axis=(1, 2))
    percentiles90 = np.percentile(_entr_raw, 90, axis=(1, 2))
    percentiles50 = np.percentile(_entr_raw, 50, axis=(1, 2))
    # %%
    ### Define threshould through otsu in percentiles?
    threshold = ski.filters.threshold_otsu(percentiles99)
    # print(threshold)

    plt.plot(percentiles99, label="99th percentile")
    plt.axhline(threshold, color="blue", label="99th perc. thresh.", ls="--")
    plt.plot(percentiles90, label="90th percentile")
    plt.plot(percentiles50, label="Median")
    plt.legend()

    plt.suptitle("Removal of slices without information")
    plt.xlabel(f"Slice")
    plt.ylabel(f"Local Entropy [bits]")
    ### format x axis to integers
    plt.xticks(np.arange(0, len(percentiles99), 1))

    ### Adjust height of figure
    plt.gcf().set_size_inches(7, 3)

    plt.savefig("3_entropy_percentiles.pdf", bbox_inches="tight")

    # %%
    ### Test case where there is no discernable threshold for entropy
    # percentiles2 = percentiles[percentiles > threshold]
    # # %%
    # threshold2 = ski.filters.threshold_otsu(percentiles2)
    # plt.plot(percentiles2, label="99percentiles")
    # plt.axhline(threshold2, color="red", label="threshold")
    # # !! otsu is greedy, it will always find a threshold
    # %%

    # Z.imgs[Z.imgs > 0.0] = 1

    # plt.imshow(Z.imgs[0])

    # %%
    from imagep.images.stack import Stack
    from imagep.processing.process import Process
    from imagep.pipeline import Pipeline
    
    self = Stack
    data = Pipeline
    issubclass(self, data)
    isinstance(Stack, Process)
    
    # %%
    ### Check history
    # Z.info
    # %%
    # Z.history

    # %%
