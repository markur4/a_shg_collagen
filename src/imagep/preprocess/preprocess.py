#
# %%
import os
from pprint import pprint

from collections import OrderedDict


import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import skimage as ski




# %%
# == Cache ===========================================================
# location = Path(".", "_cache")
location = os.path.join(
    os.path.expanduser("~"),
    ".cache",
)

### Subcache
from imagep.subcache import SubCache

MEMORY = SubCache(
    location=location,
    subcache_dir="preprocess",
    verbose=True,
    compress=9,
)


# %%
# == IMPORTING =========================================================
def from_txt(path: str, type=np.float32) -> np.ndarray:
    """Import from a txt file."""
    return np.loadtxt(path, skiprows=2).astype(type)


if __name__ == "__main__":
    t = np.float32
    # path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_6.txt"
    # img = from_txt(path, type=t)
    # print(img.min(), img.max())
    # plt.imshow(img)
    # plt.show()

    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_7.txt"
    img = from_txt(path, type=t)
    print(img.min(), img.max())
    plt.imshow(img)

    # %%
    ### Find smallest difference
    img_diff = ski.filters.sobel(img)
    print(img_diff.min(), img_diff.max())
    plt.imshow(img_diff)


# %%
# == CLASS PREPROCESSING ===============================================


class PreProcess:
    def __init__(
        self,
        path: str | Path,
        x_µm: float = 200,
        ### Pre-Process_kws
        denoise=False,
        normalize=True,
        subtract_bg: bool = False,
        subtract_bg_kws: dict = dict(method="triangle", sigma=1.5),
        scalebar_micrometer: bool |int = False,
        cache_preprocessing=True,
        verbose=True,
    ) -> None:
        """Import a z-stack from a folder. Performs normalization.

        :param path: pathlike string or object
        :type path: str | Path
        :param normalize: Wether to normalize values between 1 and 0,
            defaults to True
        :type normalize: bool, optional
        :param z_dist: Distance in µm between each image in z-direction,
            defaults to .5
        :type z_dist: float, optional
        :param width: Width of the image in µm, defaults to 200
        :type width: float, optional
        """

        self.verbose = verbose

        ### Folder of the z-stack
        self.path = Path(path)
        self._check_path()

        ### Import Images
        if self.verbose:
            print(f"=> Importing Images from {self.path_short}...")
        self.imgs: np.ndarray = self.import_imgs(self.path)
        if self.verbose:
            print("   Importing Images Done")

        ### width, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.imgs.shape[1] * self.x_µm / self.imgs.shape[2]
        self.pixel_size = self.x_µm / self.imgs.shape[2]
        self.spacing = (self.x_µm, self.y_µm)

        # == Pre-Processing ==

        ### Collect all preprocessing kws
        self.kws_preprocess = {
            "denoise": denoise,
            "normalize": normalize,
            "subtract_bg": subtract_bg,
            "subtract_bg_kws": subtract_bg_kws,
            "scalebar_micrometer": scalebar_micrometer,
        }

        ### Document History of processing steps
        self.history: OrderedDict = self._init_history(**self.kws_preprocess)

        ### Execute !
        if self.verbose:
            print(f"=> Pre-processing: {list(self.history.keys())[1:]} ...")
        if cache_preprocessing:
            if self.verbose:
                print("\tChecking Cache...")
            self.imgs = self._preprocess_cached(**self.kws_preprocess)
        else:
            self.imgs = self._preprocess(**self.kws_preprocess)
        if self.verbose:
            print("   Pre-processing Done")

    def _preprocess_cached(self, **preprocess_kws):
        """Preprocess the z-stack"""
        preprocess_cached = MEMORY.subcache(_preprocess_main)
        return preprocess_cached(self, **preprocess_kws)

    def _preprocess(self, **preprocess_kws):
        """Preprocess the z-stack"""
        return _preprocess_main(self, **preprocess_kws)

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
            OD[
                "BG Subtraction"
            ] = f"Calculated threshold (method = {kws['method']}) of blurred images (gaussian filter, sigma = {kws['sigma']}). Subtracted threshold from images and set negative values set to 0"

        if preprocess_kws["normalize"]:
            OD[
                "Normalization"
            ] = "Division by max value of every image in folder"

        if not isinstance(
            preprocess_kws["scalebar_micrometer"], (type(False), type(None))
        ):
            OD[
                "Scalebar"
            ] = f"Added scalebar of {preprocess_kws['scalebar_micrometer']} µm"

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

    @property
    def path_short(self) -> str:
        """Shortened path"""
        return str(self.path.parent.name + "/" + self.path.name)

    @staticmethod
    def _adj(s: str) -> str:
        J = 15
        return str(s + ": ").ljust(J).rjust(J + 2)

    @staticmethod
    def _info_brightness(S: np.ndarray) -> list:
        """Returns info about brightness for a Stack"""
        adj = PreProcess._adj

        return [
            adj("min, max") + f"{S.min():.1e}, {S.max():.1e}",
            adj("mean ± std") + f"{S.mean():.1e} ± {S.std():.1e}",
            adj("median (IQR)")
            + f"{np.median(S):.1e} ({np.quantile(S, .25):.1e} - {np.quantile(S, .75):.1e})",
        ]

    @property
    def _info(self) -> str:
        """String representation of the object for __repr__"""
        ### Shorten variables
        adj = self._adj
        S = self.imgs

        ### Check if background was subtracted
        bg_subtracted = str(self.kws_preprocess["subtract_bg"])

        # > Ignore background (0), or it'll skew statistics when bg is subtracted
        S_BG = S[S > 0.0]

        ### Fill info
        ID = OrderedDict()

        ID["Data"] = [
            "=== Data ===",
            adj("folder") + str(self.path.name),
            adj("dtype") + str(S.dtype),
            adj("shape") + str(S.shape),
            adj("images") + str(S.shape[0]),
        ]
        ID["Brightness"] = [
            "=== Brightness ===",
            adj("BG subtracted") + bg_subtracted,
        ] + self._info_brightness(S_BG)

        ID["Distance"] = [
            "=== Distances [µm] ===",
            adj("pixel size xy") + f"{self.pixel_size:.2f}",
            adj("x, y") + f"{self.x_µm:.2f}, {self.y_µm:.2f}",
        ]

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
        ### Remove memory address so joblib doesn't redo the preprocessing
        return f'<{self.__class__.__name__} object from image folder: "{self.path_short}">'

    #
    # == UTILS ====================================================

    @staticmethod
    def check_arguments(kws: dict, required_keys: list):
        """Check if all required keys are present in kws"""
        for k in required_keys:
            if not k in kws.keys():
                raise KeyError(f"Missing argument '{k}' in kws: {kws}")

    @property
    def stack_raw(self) -> np.ndarray:
        return self.import_imgs()

    def __iter__(self):
        return iter(self.imgs)

    def __getitem__(self, val: slice) -> np.ndarray:
        return self.imgs[val]

    #
    # == Import ========================================================

    def _check_path(self) -> None:
        """Check if path is valid"""
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    @staticmethod
    def import_imgs(path: Path) -> np.ndarray:
        """Import z-stack from a folder"""

        ### Get all txt files
        txts = list(path.glob("*.txt"))

        ### sort txts by number
        txts = sorted(txts, key=lambda x: int(x.stem.split("_")[-1]))

        ### Invert, since the first image is the bottom one
        txts = txts[::-1]

        ### Import all txt files
        imgs = []
        for txt in txts:
            imgs.append(from_txt(txt))

        ### Convert to numpy array
        imgs = np.array(imgs)

        return imgs

    #
    # == Metrics =======================================================

    def brightness_distribution(self) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        plt.hist(self.imgs.flatten(), bins=75, log=True)
        plt.show()

    #
    # == Normalize =====================================================

    def normalize(self) -> np.ndarray:
        """Normalize the z-stack"""
        return self.imgs / self.imgs.max()

    #
    # == Transforms ====================================================

    @staticmethod
    def denoise(imgs: np.ndarray) -> np.ndarray:
        ### List comprehensions are faster
        sigmas = [np.mean(ski.restoration.estimate_sigma(img)) for img in imgs]
        stack_denoised = [
            ski.restoration.denoise_nl_means(
                img,
                h=0.8 * sigma,
                sigma=sigma,
                patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                fast_mode=True,
            )
            for img, sigma in zip(imgs, sigmas)
        ]

        return np.array(stack_denoised)

    def blur(self, sigma: float = 1, normalize=True) -> np.ndarray:
        """Blur image using a thresholding method"""

        imgs = ski.filters.gaussian(self.imgs, sigma=sigma)

        ### The max value is not 1 anymore
        if normalize:
            imgs = imgs / imgs.max()

        return imgs

    #
    # == BG Subtract ===================================================

    @staticmethod
    def get_background_by_percentile(
        img: np.ndarray, percentile=10
    ) -> np.float64:
        """Defines background as percentile of the stack"""
        return np.percentile(img, percentile, axis=0)

    @staticmethod
    def get_background_by_threshold(
        img: np.ndarray, threshold=0.05
    ) -> np.float64:
        """Defines background as threshold * max value"""
        return img.max() * threshold

    @staticmethod
    def get_background(
        img: np.ndarray, method="triangle", sigma: float = None, **kws
    ) -> np.float64:
        ### Blur if sigma is given
        # > Improves thresholding by decreasing variance of bg

        ### Blur
        if not sigma is None:
            img = ski.filters.gaussian(img, sigma=sigma)

        ### Apply Filters
        if method == "otsu":
            return ski.filters.threshold_otsu(img, **kws)
        elif method == "mean":
            return ski.filters.threshold_mean(img, **kws)
        elif method == "triangle":
            return ski.filters.threshold_triangle(img, **kws)
        elif method == "percentile":
            return PreProcess.get_background_by_percentile(img, **kws)
        elif method == "threshold":
            return PreProcess.get_background_by_threshold(img, **kws)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def subtract(img: np.ndarray, value: float) -> np.ndarray:
        """subtracts value from stack and sets negative values to 0"""
        img_bg = img - value
        ### Set negative values to 0
        img_bg[img_bg < 0] = 0

        return img_bg

    @staticmethod
    def subtract_background(
        imgs: np.ndarray,
        method: str,
        sigma: float,
        bg_per_img=False,
    ) -> np.ndarray:
        if bg_per_img:
            imgs_bg = np.zeros(imgs.shape)
            for i, img in enumerate(imgs):
                bg = PreProcess.get_background(img, method=method, sigma=sigma)
                imgs_bg[i] = PreProcess.subtract(img, value=bg)
            return imgs_bg

        else:
            bg = PreProcess.get_background(imgs, method=method, sigma=sigma)
            return PreProcess.subtract(imgs, value=bg)

    #
    # == Annotations ===================================================

    def _add_scalebar_to_img(
        self,
        img: np.ndarray = None,
        I: int = None,
        μm: int = 10,
        thickness_μm=3,
    ) -> np.ndarray:
        """Add scalebar to an image selected by its index within the
        self.stack"""

        ### Get Image, if not given
        if img is None:
            if I is None:
                raise ValueError("Either img or index (I) must be given")
            img = self.imgs[I]

        ### Convert µm to pixels
        len_px = int(round(μm / self.pixel_size))
        thickness_px = int(round(thickness_μm / self.pixel_size))

        ### Define Scalebar as an array
        # > Color is derived from img colormap
        bar_color = self.imgs.max() * 1
        scalebar = np.zeros((thickness_px, len_px))
        scalebar[:, :] = bar_color

        ### Add Frame around scalebar with two pixels thickness
        frame_color = self.imgs.max() * 0.9
        t = 3  # Thickness of frame in pixels
        scalebar[0 : t + 1, :] = frame_color
        scalebar[-t:, :] = frame_color
        scalebar[:, 0 : t + 1] = frame_color
        scalebar[:, -t:] = frame_color

        ### Define padding from bottom right corner
        pad_x = int(self.imgs.shape[2] * 0.05)
        pad_y = int(self.imgs.shape[1] * 0.05)

        ### Add scalebar to the bottom right of the image
        # !! Won't work if nan are at scalebar position
        img[-pad_y - thickness_px : -pad_y, -pad_x - len_px : -pad_x] = scalebar
        return img

    def annotate_barsize(
        self,
        μm: int = 10,
        thickness_µm=3,
        color="black",
    ) -> np.ndarray:
        """Adds length of scalebar to image as text during plotting"""

        text = f"{μm} µm"
        # offsetbox = TextArea(text, minimumdescent=False)

        pad_x = int(self.imgs.shape[2] * 0.05)
        pad_y = int(self.imgs.shape[1] * 0.05)

        x = self.imgs.shape[2] - pad_x - thickness_µm / self.pixel_size * 2
        y = self.imgs.shape[1] - pad_y - thickness_µm / self.pixel_size

        coords = "data"

        plt.annotate(
            text,
            xy=(x, y),
            xycoords=coords,
            xytext=(x, y),
            textcoords=coords,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )

    def add_scalebar(
        self,
        all=False,
        Indexes: list = [0],
        μm: int = 10,
        thickness_μm=3,
    ) -> np.ndarray:
        """Adds scalebar to images in stack. By default, adds only to
        the first image, but can be changed with indexes."""

        if all:
            Indexes = range(self.imgs.shape[0])

        imgs = self.imgs
        for i in Indexes:
            imgs[i] = self._add_scalebar_to_img(
                I=i,
                μm=μm,
                thickness_μm=thickness_μm,
            )
        return imgs

    #
    # == I/O ===========================================================

    def save(self, path: str | Path) -> None:
        """Save the z-stack to a folder"""
        np.save(path, self.imgs)

    def load(self, path: str | Path) -> None:
        """Load the z-stack from a folder"""
        self.imgs = np.load(path)

    #
    # == Basic Visualization ===========================================

    @staticmethod
    def _mip(
        stack: np.ndarray,
        axis: int = 0,
        show=True,
        return_array=False,
        savefig: str = "mip.png",
        colormap: str = "gist_ncar",
    ) -> np.ndarray | None:
        """Maximum intensity projection across certain axis"""
        
        mip = stack.max(axis=axis)

        if show:
            plt.imshow(
                mip,
                cmap=colormap,
                interpolation="none",
            )
            plt.show()

        if savefig:
            plt.imsave(fname=savefig, arr=mip, cmap=colormap, dpi=300)

        if return_array:
            return mip

    def mip(self, **mip_kws) -> np.ndarray | None:
        return self._mip(self.imgs, **mip_kws)


#
# !!
# == preprocess_main ===================================================
def _preprocess_main(
    preprocess_object: PreProcess,
    **preprocess_kws,
) -> np.ndarray:
    self = preprocess_object

    ### Denoise
    if self.verbose:
        print("\tDenoising...")
    if preprocess_kws["denoise"]:
        self.imgs = self.denoise(self.imgs)

    ### Subtract Background
    if preprocess_kws["subtract_bg"]:
        kws = preprocess_kws.get("subtract_bg_kws", dict())
        self.check_arguments(kws, ["method", "sigma"])

        # > Subtract
        self.imgs = self.subtract_background(
            self.imgs,
            **preprocess_kws["subtract_bg_kws"],
        )

    ### Normalize
    if preprocess_kws["normalize"]:
        self.imgs = self.normalize()

    ### Add Scalebar
    # > If scalebar_µm is not None
    if not preprocess_kws["scalebar_micrometer"] in [0, False, None]:
        self.imgs = self.add_scalebar(
            Indexes=[0], µm=preprocess_kws["scalebar_micrometer"]
        )
    return self.imgs


# %%
# == TESTS ============================================================

if __name__ == "__main__":
    ### Import from a txt file.
    # > Rough
    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    kws = dict(
        # z_dist=10 * 0.250,  # > stepsize * 0.250 µm
        x_µm=1.5
        # > fast axis amplitude 1.5 V * calibration 115.4 µm/V
        * 115.4,
    )
    # > Detailed
    # path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    # kws = dict(
    #     # z_dist=2 * 0.250,  # > stepsize * 0.250 µm
    #     x_µm=1.5
    #     * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    # )
    Z = PreProcess(
        path=path,
        subtract_bg=False,
        scalebar_micrometer=50,
        **kws,
    )
    Z.imgs.shape
    Z.imgs.max()
    # plt.imshow(zstack.stack[0])

    # %%
    ### Check history
    Z.info

    # %%
    MEMORY.list_objects()

    # %%
    Z.history

    # %%
    Z.kws_preprocess

    # %%
    Z.kws_preprocess["scalebar_micrometer"]

    # %%
    bool(Z.kws_preprocess.get("scalebar_micrometer"))

    # %%
    isinstance(
        Z.kws_preprocess["scalebar_micrometer"], (type(False), type(None))
    )

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
    #%%
    Z_d.info
    
    #%%
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
