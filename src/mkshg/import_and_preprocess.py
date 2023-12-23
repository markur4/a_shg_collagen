#
# %%

from pprint import pprint

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D


import skimage.restoration as skr


# print(pv.Report())

# from pyvista import examples


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
        z_dist: float = 0.5,
        x_µm: float = 200,
        normalize=True,
        denoise=False,
        scalebar_µm: int | None = 50,
        background_subtract: float = None,
        # fill_z=True,
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
        self.stack = self.import_stack()

        ### Z-distance
        self.z_dist = z_dist

        ### width, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.stack.shape[1] * self.x_µm / self.stack.shape[2]
        self.z_µm = self.stack.shape[0] * self.z_dist
        self.pixel_size = self.x_µm / self.stack.shape[2]
        self.spacing = (self.x_µm, self.y_µm, self.z_µm)

        ### Denoise
        if denoise:
            self.stack = self.denoise()

        ### Background subtract
        assert isinstance(background_subtract, (float, int)) or background_subtract is None, (
            f"background_subtract must be float or int, not {type(background_subtract)}"
        )
        if background_subtract:
            background = self.get_background_by_threshold(
                threshold=background_subtract
            )
            self.stack = self.subtract(background)

        ### Normalize
        if normalize:
            self.stack = self.normalize()

        ### Add Scalebar
        if scalebar_µm:
            self.stack = self.add_scalebar(width_μm=scalebar_µm)


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
    # == Pre-Processing ================================================

    def normalize(self) -> np.ndarray:
        """Normalize the z-stack"""
        return self.stack / self.stack.max()

    def denoise(self) -> np.ndarray:
        ### List comprehensions are faster
        sigmas = [np.mean(skr.estimate_sigma(img)) for img in self.stack]
        stack_denoised = [
            skr.denoise_nl_means(
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

    def get_background_by_percentile(self, percentile=10) -> np.float64:
        """Defines background as percentile of the stack"""
        return np.percentile(self.stack, percentile, axis=0)

    def get_background_by_threshold(self, threshold=0.05) -> np.float64:
        """Defines background as threshold * max value"""
        return self.stack.max() * threshold

    def subtract(self, background: float):
        """subtracts value from stack and sets negative values to 0"""
        stack_bg = self.stack - background
        ### Set negative values to 0
        stack_bg[stack_bg < 0] = 0

        return stack_bg

    def add_scalebar(self, width_µm=50, thickness_µm=3) -> np.ndarray:
        """Burns a scalebar into the bottom right corner of the image"""
        pixelwidth = int(round(width_µm / self.pixel_size))
        pixelthickness = int(round(thickness_µm / self.pixel_size))

        pad_x = int(self.stack.shape[2] * 0.05)
        pad_y = int(self.stack.shape[1] * 0.05)

        value = self.stack.max() * 0.8

        ### Add scalebar to the bottom image
        stack = self.stack
        stack[
            0, -pad_y - pixelthickness : -pad_y, -pad_x - pixelwidth : -pad_x
        ] = value

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
    def mip(self, axis: int = 0, show=True, ret=False) -> np.ndarray:
        """Maximum intensity projection across certain axis"""
        p = self.stack.max(axis=axis)

        if show:
            plt.imshow(p, cmap="gist_ncar")

        if show:
            plt.show()
        if ret:
            return p



if __name__ == "__main__":
    pass
    # %%                                             
    ### Import from a txt file.
    # > Rough
    # path =
    # "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1
    # healthy z-stack rough/"
    # kws = dict(
    #     z_dist=10 * 0.250,  # stepsize * 0.250 µm
    #     x_µm=1.5 * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    # )
    # > Detailed
    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    kws = dict(
        z_dist=2 * 0.250,  # stepsize * 0.250 µm
        x_µm=1.5 * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    )
    Z = PreProcess(path, **kws)
    Z.stack.shape
    Z.stack.max()
    # plt.imshow(zstack.stack[0])

    # %%
    Z.mip(axis=0)  # ' z-axis
    # %%
    Z.mip(axis=1)  # ' x-axis

    # %%
    Z.mip(axis=2)  # ' y-axis
    # %%
    print(Z.x_µm)
    print(Z.y_µm)
    print(Z.z_µm)
    print(Z.pixel_size)
    print(Z.spacing)

    # %%
    #:: Denoising makes background subtraction better
    # Z_d = ZStack(path, denoise=True, **kws)
    Z_d_bg = PreProcess(
        path,
        denoise=True,
        background_subtract=0.06,  # > In percent of max brightness
        **kws,
    )

    # %%
    Z.brightness_distribution()
    # Z_d.brightness_distribution()
    Z_d_bg.brightness_distribution()

    # %%
    #:: Denoising preserves textures!
    mip = Z.mip(ret=True)
    # mip_d = Z_d.mip(ret=True)
    mip_d_bg = Z_d_bg.mip(ret=True)

    # %%
    sns.boxplot(
        [
            mip.flatten(),
            # mip_d.flatten(),
            mip_d_bg.flatten(),
        ],
        showfliers=False,
    )
