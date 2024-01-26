"""A Class for basic funtionalities of images.
- Import
- Pixel scale to µm
- Add Scalebar
- Plotting mip
- 
"""
# %%
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

# > Local
import imagep.imgs.importutils as importutils
import imagep.utils.scalebar as scalebar
import imagep.utils.utils as ut


# %%
# == Class Imgs ========================================================


class Imgs:
    def __init__(
        self,
        path: str | Path,
        x_µm: float = 200.0,
        scalebar_microns: int = 10,
        verbose: bool = True,
    ) -> None:
        """Basic Image-stack funcitonalities.
        :param path: pathlike string or object
        :type path: str | Path
        :param x_µm: Total width of image in µm, defaults to 200
        :type x_µm: float, optional
        :param scalebar_microns: Length of scalebar in µm. This won't
            add a scalebar, it's called once needed, defaults to
            10
        :type scalebar_microns: int, optional
        """
        self.verbose = verbose
        self.scalebar_microns = scalebar_microns

        ### Folder of the image stack
        self.path = Path(path)
        self._check_path()

        ### Import Images
        if self.verbose:
            print(f"=> Importing Images from {self.path_short}...")
        self.imgs: np.ndarray = self.import_imgs(self.path)
        if self.verbose:
            print("   Importing Images Done")

        ### Totalwidth, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.imgs.shape[1] * self.x_µm / self.imgs.shape[2]
        self.pixel_size = self.x_µm / self.imgs.shape[2]
        self.spacing = (self.x_µm, self.y_µm)

    # == Properties =====================================================

    @property
    def path_short(self) -> str:
        """Shortened path"""
        return str(self.path.parent.name + "/" + self.path.name)

    #
    # == Access Images =================================================

    @property
    def stack_raw(self) -> np.ndarray:
        return self.import_imgs()

    def __iter__(self):
        return iter(self.imgs)

    def __getitem__(self, val: slice) -> np.ndarray:
        return self.imgs[val]

    #
    # == Import Images =================================================

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
            imgs.append(importutils.from_txt(txt))

        ### Convert to numpy array
        imgs = np.array(imgs)

        return imgs

    #
    # == Scalebar ======================================================

    def burn_scalebar(
        self,
        all: bool = True,
        indexes: list = [0],
        xy_pad: tuple[float] = (0.05, 0.05),
        thickness: int = 3,
    ) -> None:
        """Burns scalebar to images in stack. By default, adds only to
        the first image, but can be changed with indexes."""

        if all:
            indexes = range(self.imgs.shape[0])

        # imgs_result = np.zeros(self.imgs.shape, dtype=self.imgs.dtype)
        for i in indexes:
            self.imgs[i] = scalebar.burn_scalebar_to_img(
                img=self.imgs[i],
                microns=self.scalebar_microns,
                pixel_size=self.pixel_size,
                thickness=thickness,
                xy_pad=xy_pad,
                bar_color=self.imgs.max(),
                frame_color=self.imgs.max() * 0.9,
            )

    def annot_micronlength_into_plot(
        self,
        img: np.ndarray = None,
        xy_pad: tuple[float] = (0.05, 0.05),
        thickness: int = 3,
        color="white",
    ) -> None:
        ### img required to find correct position
        img = self.imgs[0] if img is None else img
        
        scalebar.annot_micronlength_into_plot(
            img=img,
            pixel_size=self.pixel_size,
            microns=self.scalebar_microns,
            xy_pad=xy_pad,
            thickness=thickness,
            color=color,
        )


    #
    # == I/O ===========================================================

    def save(self, fname: str | Path) -> None:
        """Save the z-stack to a folder"""
        np.save(fname, self.imgs)

    def load(self, fname: str | Path) -> None:
        """Load the z-stack from a folder"""
        self.imgs = np.load(fname)

    #
    # == Plots =========================================================

    def mip(self, **mip_kws) -> np.ndarray | None:
        return ut.mip(self.imgs, **mip_kws)

    @staticmethod
    def plot_brightness_distribution(
        imgs: np.ndarray, bins=75, log=True
    ) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        plt.hist(imgs.flatten(), bins=bins, log=log)
        plt.show()
