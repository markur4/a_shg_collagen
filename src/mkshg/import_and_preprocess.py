#
# %%

from pprint import pprint

from collections import OrderedDict


import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D


from skimage import restoration
from skimage import filters

# print(pv.Report())

# from pyvista import examples


# %%
# == Cache ===========================================================
location = "./_cachedir"

### Raw joblib
# from joblib import Memory
# > https://joblib.readthedocs.io/en/latest/memory.html
# ### If the folder does not exist, create it, it exists, delete it
# if not Path(location).exists():
#     Path(location).mkdir()


# memory = Memory(location, verbose=0)

### Subcache
# from mkshg.subcache import SubCache
# memory = SubCache(
#     location=location,
#     subcache_dir="preprocess",
#     verbose=0,
# )


# %%
def from_txt(path: str) -> np.ndarray:
    """Import from a txt file."""
    return np.loadtxt(path, skiprows=2)


if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/Image3_6.txt"
    img = from_txt(path)
    plt.imshow(img)


# %%
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
        scalebar_micrometer: int = 10,
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

        ### Folder of the z-stack
        self.path = Path(path)
        self._check_path()

        ### Import z-stack
        self.stack: np.ndarray = self.import_stack()

        ### width, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.stack.shape[1] * self.x_µm / self.stack.shape[2]
        self.pixel_size = self.x_µm / self.stack.shape[2]
        self.spacing = (self.x_µm, self.y_µm)

        # == Execute Pre-Processing ==

        ### Collect all preprocessing kws
        self.preprocess_kws = {
            "denoise": denoise,
            "normalize": normalize,
            "subtract_bg": subtract_bg,
            "subtract_bg_kws": subtract_bg_kws,
            "scalebar_micrometer": scalebar_micrometer,
        }

        ### Document History of processing steps
        self._history: OrderedDict = self._init_history(**self.preprocess_kws)

        ### Execute Pre-Processing!
        self.stack = self._preprocess(**self.preprocess_kws)

    def _preprocess(self, **preprocess_kws):
        """Preprocess the z-stack"""

        ### Denoise
        if preprocess_kws["denoise"]:
            ### Cached
            # denoise_cached = memory.cache(self.denoise)
            # self.stack = denoise_cached()
            self.stack = self.denoise()

        ### Subtract Background
        if preprocess_kws["subtract_bg"]:
            kws = preprocess_kws.get("subtract_bg_kws", dict())
            self.check_arguments(kws, ["method", "sigma"])

            # > Subtract
            self.stack = self.subtract_background(
                **preprocess_kws["subtract_bg_kws"]
            )

        ### Normalize
        if preprocess_kws["normalize"]:
            self.stack = self.normalize()

        ### Add Scalebar
        # > If scalebar_µm is not None
        if not isinstance(
            preprocess_kws["scalebar_micrometer"], (type(False), type(None))
        ):
            # self.stack = self.add_scalebar(
            #     µm=preprocess_kws["scalebar_micrometer"]
            # )

            self.stack = self.add_scalebar(
                Indexes=[0], µm=preprocess_kws["scalebar_micrometer"]
            )
        return self.stack

    #
    # === HISTORY ====================================================

    def _init_history(self, **preprocess_kws) -> OrderedDict:
        """Update the history of processing steps"""

        OD = OrderedDict()

        if preprocess_kws["denoise"]:
            OD["Denoised"] = "Non-local means"

        if preprocess_kws["subtract_bg"]:
            kws = preprocess_kws.get("subtract_bg_kws", dict())
            OD[
                "BG subtracted"
            ] = f"Calculated threshold (method = {kws['method']}) of blurred images (gaussian filter, sigma = {kws['sigma']}). Subtracted threshold from images and set negative values set to 0"

        if preprocess_kws["normalize"]:
            OD["Normalized"] = "Division by max value"

        if not isinstance(
            preprocess_kws["scalebar_micrometer"], (type(False), type(None))
        ):
            OD[
                "Scalebar Added"
            ] = f"Added scalebar of {preprocess_kws['scalebar_micrometer']} µm"

        return OD

    @property
    def history(self) -> dict:
        return dict(self._history)

    def _history_to_str(self) -> str:
        """Returns the history of processing steps"""

        string = ""
        for i, (k, v) in enumerate(self._history.items()):
            string += f"  {i+1}. {k}: ".ljust(23)
            string += v + "\n"
        return string

    def print_history(self) -> None:
        """Print the history of processing steps"""

        print(self._history_to_str())

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
        return self.import_stack()

    def __iter__(self):
        return iter(self.stack)

    def __getitem__(self, val: slice) -> np.ndarray:
        return self.stack[val]

    #
    # == __repr__ ======================================================

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
        S = self.stack

        ### Check if background was subtracted
        bg_subtracted = str(self.preprocess_kws["subtract_bg"])

        # > Ignore background (0), or it'll skew statistics when bg is subtracted
        S_BG = S[S > 0.0]

        ### Fill info
        ID = OrderedDict()

        ID["Data"] = [
            "=== Data ===",
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
        for k, v in info.items():
            string += "\n".join(v) + "\n\n"

        return string

    def __repr__(self) -> str:
        return self._info_to_str(self._info)

    #
    # == Import ========================================================

    def _check_path(self) -> None:
        """Check if path is valid"""
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    def import_stack(self) -> np.ndarray:
        """Import z-stack from a folder"""

        ### Get all txt files
        txts = list(self.path.glob("*.txt"))

        ### sort txts by number
        txts = sorted(txts, key=lambda x: int(x.stem.split("_")[-1]))

        ### Invert, since the first image is the bottom one
        txts = txts[::-1]

        ### Import all txt files
        stack = []
        for txt in txts:
            stack.append(from_txt(txt))

        ### Convert to numpy array
        stack = np.array(stack)

        return stack

    #
    # == Metrics =======================================================

    def brightness_distribution(self) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        plt.hist(self.stack.flatten(), bins=75, log=True)
        plt.show()

    #
    # == Normalize =====================================================

    def normalize(self) -> np.ndarray:
        """Normalize the z-stack"""
        return self.stack / self.stack.max()

    #
    # == Transforms ====================================================

    def denoise(self) -> np.ndarray:
        ### List comprehensions are faster
        sigmas = [
            np.mean(restoration.estimate_sigma(img)) for img in self.stack
        ]
        stack_denoised = [
            restoration.denoise_nl_means(
                img,
                h=0.8 * sigma,
                sigma=sigma,
                patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                fast_mode=True,
            )
            for img, sigma in zip(self.stack, sigmas)
        ]

        return np.array(stack_denoised)

    def blur(self, sigma: float = 1, normalize=True) -> np.ndarray:
        """Blur image using a thresholding method"""

        stack = filters.gaussian(self.stack, sigma=sigma)

        ### The max value is not 1 anymore
        if normalize:
            stack = stack / stack.max()

        return stack

    #
    # == BG Subtract ===================================================

    @staticmethod
    def get_background_by_percentile(
        stack: np.ndarray, percentile=10
    ) -> np.float64:
        """Defines background as percentile of the stack"""
        return np.percentile(stack, percentile, axis=0)

    @staticmethod
    def get_background_by_threshold(
        stack: np.ndarray, threshold=0.05
    ) -> np.float64:
        """Defines background as threshold * max value"""
        return stack.max() * threshold

    def get_background(
        self, method="triangle", sigma: float = None, **kws
    ) -> np.float64:
        ### Blur if sigma is given
        # > Improves thresholding by decreasing variance of bg

        ### Get stack
        stack = self.stack

        ### Blur
        if not sigma is None:
            stack = filters.gaussian(stack, sigma=sigma)

        ### Apply Filters
        if method == "otsu":
            return filters.threshold_otsu(stack, **kws)
        elif method == "mean":
            return filters.threshold_mean(stack, **kws)
        elif method == "triangle":
            return filters.threshold_triangle(stack, **kws)
        elif method == "percentile":
            return self.get_background_by_percentile(stack, **kws)
        elif method == "threshold":
            return self.get_background_by_threshold(stack, **kws)
        else:
            raise ValueError(f"Unknown method: {method}")

    def subtract(self, value: float) -> np.ndarray:
        """subtracts value from stack and sets negative values to 0"""
        stack_bg = self.stack - value
        ### Set negative values to 0
        stack_bg[stack_bg < 0] = 0

        return stack_bg

    def subtract_background(self, method: str, sigma: float) -> np.ndarray:
        background = self.get_background(method=method, sigma=sigma)
        return self.subtract(background)

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
            img = self.stack[I]

        ### Convert µm to pixels
        len_px = int(round(μm / self.pixel_size))
        thickness_px = int(round(thickness_μm / self.pixel_size))

        ### Define Scalebar as an array
        # > Color is derived from img colormap
        bar_color = self.stack.max() * 1
        scalebar = np.zeros((thickness_px, len_px))
        scalebar[:, :] = bar_color

        ### Add Frame around scalebar with two pixels thickness
        frame_color = self.stack.max() * 0.9
        t = 3  # Thickness of frame in pixels
        scalebar[0 : t + 1, :] = frame_color
        scalebar[-t:, :] = frame_color
        scalebar[:, 0 : t + 1] = frame_color
        scalebar[:, -t:] = frame_color

        ### Define padding from bottom right corner
        pad_x = int(self.stack.shape[2] * 0.05)
        pad_y = int(self.stack.shape[1] * 0.05)

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

        pad_x = int(self.stack.shape[2] * 0.05)
        pad_y = int(self.stack.shape[1] * 0.05)

        x = self.stack.shape[2] - pad_x - thickness_µm / self.pixel_size * 2
        y = self.stack.shape[1] - pad_y - thickness_µm / self.pixel_size

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
            Indexes = range(self.stack.shape[0])

        stack = self.stack
        for i in Indexes:
            stack[i] = self._add_scalebar_to_img(
                I=i,
                μm=μm,
                thickness_μm=thickness_μm,
            )
        return stack

    #
    # == I/O ===========================================================

    def save(self, path: str | Path) -> None:
        """Save the z-stack to a folder"""
        np.save(path, self.stack)

    def load(self, path: str | Path) -> None:
        """Load the z-stack from a folder"""
        self.stack = np.load(path)

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
        return self._mip(self.stack, **mip_kws)


if __name__ == "__main__":
    pass
    # %%
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
    Z.stack.shape
    Z.stack.max()
    # plt.imshow(zstack.stack[0])

    # %%
    ### Check history
    Z.history

    # %%
    Z.print_history()

    # %%
    Z.preprocess_kws

    # %%
    Z.preprocess_kws["scalebar_micrometer"]

    # %%
    bool(Z.preprocess_kws.get("scalebar_micrometer"))

    # %%
    isinstance(
        Z.preprocess_kws["scalebar_micrometer"], (type(False), type(None))
    )

    # %%
    ### Check __repr__
    Z._info

    # %%
    Z_bg = PreProcess(path, subtract_bg=True, **kws)

    # %%
    Z

    # %%
    Z_bg

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
    #:: what's better to flatten background: denoise or blurring?

    S = Z_d.blur(sigma=1)
    plt.imshow(S[7])

    # %%
    histkws = dict(bins=200, log=False, alpha=0.4)

    plt.hist(Z.stack.flatten(), label="raw", **histkws)
    plt.hist(S.flatten(), label="blur", **histkws)
    plt.legend()

    # %%
    plt.hist(Z.stack.flatten(), label="raw", **histkws)
    plt.hist(Z_d.stack.flatten(), label="denoise", **histkws)
    plt.legend()

    # %%
    plt.hist(S.flatten(), label="blur", **histkws)
    plt.hist(Z_d.stack.flatten(), label="denoise", **histkws)
    plt.legend()

    # Z.brightness_distribution()
    # Z_d.brightness_distribution()
    # Z_d_bg.brightness_distribution()

    # %%
    print(filters.threshold_triangle(Z.stack))
    print(filters.threshold_triangle(S))
    print(filters.threshold_triangle(Z_d.stack))

    # %%
    #:: Denoising preserves textures!
    mip = Z.mip(ret=True)
    # mip_d = Z_d.mip(ret=True)
    # mip_d_bg = Z_d_bg.mip(ret=True)

    # %%
    sns.boxplot(
        [
            mip.flatten(),
            # mip_d.flatten(),
            mip_d_bg.flatten(),
        ],
        showfliers=False,
    )
