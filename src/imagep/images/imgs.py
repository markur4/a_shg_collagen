""" The first interface for interacting with imageP. A Class providing
basic tools for imported images. Her we include:
- Calling main import function
- Pixel scale to µm
- Add Scalebar
- Simple visualizations
"""
# %%

from typing import Self, TYPE_CHECKING

import copy

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# > Local
import imagep._rc as rc
import imagep.images.importtools as importtools
from imagep.images.raw import ImgsRaw
import imagep._plots.scalebar as scaleb
import imagep._plots.imageplots as imageplots
import imagep._utils.utils as ut

if TYPE_CHECKING:
    from imagep.images.imgs import Imgs
# from imagep.utils.transforms import Transform


# %%
# == Class Imgs ========================================================


class Imgs(ImgsRaw):
    """Interface for handling image types."""

    def __init__(
        self,
        ### kws for ImgsRaw:
        data: str | Path | np.ndarray | list[np.ndarray] | Self = None,
        verbose: bool = True,
        dtype: np.dtype = rc.DTYPE_DEFAULT,
        ### kws for Imgs
        x_µm: float = 200.0,
        scalebar_microns: int = 10,
    ):
        """Basic Image-stack functionalities.
        :param path: pathlike string or object
        :type path: str | Path
        :param x_µm: Total width of image in µm, defaults to 200
        :type x_µm: float, optional
        :param scalebar_microns: Length of scalebar in µm. This won't
            add a scalebar, it's called once needed, defaults to
            10
        :type scalebar_microns: int, optional
        """
        # todo: Add a class for images of different sizes
        # todo: Make a colored image class

        # ### Import Images
        # super().__init__(data, dtype, verbose)

        ### Call super().__init__(), OR retrieve attributes from instance
        self._attribute_transfer(data, dtype, verbose)

        ### Total width, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.imgs.shape[1] * self.x_µm / self.imgs.shape[2]
        self.pixel_size = self.x_µm / self.imgs.shape[2]
        self.spacing = (self.x_µm, self.y_µm)

        # == Other ==
        ### Define scalebar length here, required by e.g. mip
        self.scalebar_microns = scalebar_microns

    # == Import from parent Instance ===================================
    #
    def _attribute_transfer(self, data: Self, dtype: np.dtype, verbose: bool):
        """Import images from another Imgs instance. This will transfer
        all attributes from the Imgs instance. Methods are transferred
        by inheritance, because we want the option to import images at
        every stage of the processing pipeline."""

        ### Transfer all attributes, if instance is passed
        if isinstance(data, type(self)):
            super().import_from_instance(instance=data, verbose=verbose)
        # > If not, call the parent __init__ method
        else:
            super().__init__(data, dtype, verbose)

    #
    # == Scalebar ======================================================

    def burn_scalebars(
        self,
        # slice: str | int | list[int] | tuple[int] = "all",
        xy_pad: tuple[float] = (0.05, 0.05),
        thickness_px: int = 20,
        text_color: tuple[int] | str = None,
        inplace: bool = False,
    ) -> np.ndarray:
        """Burns scalebar to images in stack. By default, adds only to
        the first image, but can be changed with indexes.
        :param imgs: Images to burn scalebar to, defaults to None
        :type imgs: np.ndarray, optional
        :param slice: Part of images to burn scalebar to. Options
        are `"all"`, `int`, `list[int]` or `tuple(start,stop,step)`,
        defaults to "all"
        :type slice: str | int | tuple[int], optional
        :param xy_pad: Distance of scalebar from bottom left corner in
        percent of image width and height, defaults to (0.05, 0.05)
        :type xy_pad: tuple[float], optional
        :param thickness: Thickness of scalebar in pixels, defaults to
        3
        :type thickness: int, optional
        :param text_color: Color of text. Either a string ("white") or a
        tuple of greyscale/RGB values, e.g. (255)/(255,255,255). If None, will use the
        brightest pixel of the image, defaults to None
        :type text_color: str | tuple[int], optional
        """
        imgs = self.imgs if inplace else self.imgs.copy()

        imgs = scaleb.burn_scalebars(
            imgs=imgs,
            # slice=slice,
            microns=self.scalebar_microns,
            pixel_size=self.pixel_size,
            thickness_px=thickness_px,
            xy_pad=xy_pad,
            bar_color=imgs.max(),
            frame_color=imgs.max() * 0.9,
            text_color=text_color,
        )
        return imgs

    #
    # == Plots =========================================================

    def imshow(
        self,
        # slice: str | int | list[int] | tuple[int] = "all",
        cmap: str = "gist_ncar",
        max_cols: int = 2,
        scalebar=True,
        scalebar_kws: dict = dict(),
        colorbar=True,
        fname: bool | str = False,
        **imshow_kws,
    ) -> None:
        """Show the images"""

        ### Make copy to ensure
        _imgs = self.imgs.copy()

        ### Update KWS
        scalebar_KWS = dict(
            microns=self.scalebar_microns,
            pixel_size=self.pixel_size,
        )
        scalebar_KWS.update(scalebar_kws)

        ### Update kwargs
        KWS = dict(
            imgs=_imgs,
            # slice=slice,
            max_cols=max_cols,
            cmap=cmap,
            scalebar=scalebar,
            scalebar_kws=scalebar_KWS,
            colorbar=colorbar,
        )
        KWS.update(imshow_kws)

        ### MAKE IMAGE
        fig, axes = imageplots.imshow(**KWS)

        ### Add Ax titles
        for i, ax in enumerate(axes.flat):
            if i >= len(self.imgs):
                break
            # > get correct index if sliced

            _i = self._slice_indices[i] if self._slice else i

            img = _imgs[i]  # > retrieve image
            axtit = (
                f"Image {i+1}/{len(self.imgs)} (i={_i}/{self._num_imgs-1})"
                f"    {img.shape[0]}x{img.shape[1]}  {img.dtype}"
                # f"\nmin={form(img.min())}  mean={form(img.mean())}  max={form(img.max())}"
            )
            ax.set_title(axtit, fontsize=10)

        ### Fig title
        tit = f"{self.path_short}\n - {self._num_imgs} Total images"
        if self._slice:
            tit += f"; Sliced to {len(_imgs)} image(s) (i=[{self._slice}])"

        ### Get number of rows in axes
        bbox_y = 1.05 if axes.shape[0] <= 2 else 1.01
        fig.suptitle(tit, ha="left", x=0.01, y=bbox_y, fontsize=12)

    def mip(self, scalebar=True, **mip_kws) -> np.ndarray | None:
        ### Make copy in case of burning in scalebar
        _imgs = self.imgs.copy()
        if scalebar:
            # > Put scalebar on first image only
            _imgs[0] = self.burn_scalebars()[0]

        mip = imageplots.mip(imgs=_imgs, **mip_kws)
        plt.show()
        return mip

    def plot_brightness_distribution(self, bins=75, log=True) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        plt.hist(self.imgs, bins=bins, log=log)
        plt.show()


#
# !! ===================================================================
# %%
# == Testdata ==========================================================
if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Imgs(data=path, verbose=True, x_µm=1.5 * 115.4)
    I = 6

    # %%
    # Z.imgs.tolist()


# %%
def _test_import_from_types(Z, I=6):
    # > Import from Path
    Z1 = Imgs(data=path, verbose=True, x_µm=1.5 * 115.4)
    Z1[I].imshow()

    # > Import from np.ndarray
    Z2 = Imgs(data=Z.imgs, verbose=True, x_µm=1.5 * 115.4)
    Z2[I].imshow()

    # > Import from list of np.ndarrays
    Z3 = Imgs(data=[im for im in Z.imgs], verbose=True, x_µm=1.5 * 115.4)
    Z3[I].imshow()

    # > Import from self
    Z4 = Imgs(data=Z, verbose=True, x_µm=1.5 * 115.4)
    Z4[I].imshow()


if __name__ == "__main__":
    _test_import_from_types(Z=Z, I=I)


# %%
def _test_imshow_method(Z):
    kws = dict(
        max_cols=2,
        scalebar=True,
    )

    Z[0].imshow(**kws)
    Z[0, 3, 5, 19].imshow(**kws)
    Z[1:3].imshow(**kws)
    Z[0:10:3].imshow(**kws)
    Z[0:10:2].imshow(**kws)
    Z[[1, 3, 6, 7, 8, 9, 15]].imshow(**kws)
    plt.show()  # >   show last one

    ### Check if scalebar is not burned it
    plt.imshow(Z.imgs[0])
    plt.suptitle("no scalebar should be here")
    plt.show()


if __name__ == "__main__":
    _test_imshow_method(Z)
