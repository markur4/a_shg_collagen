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

from pprint import pprint

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# > Local
import imagep._rc as rc

# import imagep.images.importtools as importtools
# from imagep.images.imgs_import import ImgsImport
from imagep.images.imgs_meta import ImgsMeta
import imagep._plots.scalebar as scalebar
import imagep._plots.imageplots as imageplots
import imagep._plots.dataplots as dataplots
import imagep._utils.utils as ut

if TYPE_CHECKING:
    from imagep.images.imgs import Imgs
# from imagep.utils.transforms import Transform


# %%
# ======================================================================
# == Class Imgs ========================================================
class Imgs(ImgsMeta):
    """Interface for handling images
    - Adds experimental information to Images
    """

    def __init__(
        self,
        ### ImgsImport kws:
        data: str | Path | np.ndarray | list[np.ndarray] | Self = None,
        verbose: bool = True,
        ### ImgsMeta kws:
        pixel_length: float | list[float] = None,
        unit: str = "µm",
        scalebar_length: int = None,  # > in (micro)meter
        ### fileimport_kws
        fname_pattern: str = "",
        fname_extension: str = "",
        sort: bool = True,
        imgname_position: int | list[int] = 0,
        invertorder: bool = True,
        dtype: np.dtype = rc.DTYPE,
        ### kws for importfunction, like skiprows
        **importfunc_kws,
    ):
        """Basic Image-stack functionalities.
        - Block super()__init__ if to avoid re-loading images


        :param data: pathlike string or object
        :type data: str | Path | np.ndarray | list[np.ndarray] | Self
        :param x_µm: Total width of image in µm, defaults to 200
        :type x_µm: float, optional
        :param scalebar_length: Length of scalebar in µm. This won't
            add a scalebar, it's called once needed, defaults to
            10
        :type scalebar_length: int, optional
        """
        ### GET ATTRIBUTES
        # > Collect kws
        import_kws = dict(data=data, verbose=verbose)
        meta_kws = dict(
            pixel_length=pixel_length,
            unit=unit,
            scalebar_length=scalebar_length,
        )
        fileimport_kws = dict(
            imgname_position=imgname_position,
            fname_pattern=fname_pattern,
            fname_extension=fname_extension,
            sort=sort,
            invertorder=invertorder,
            dtype=dtype,
            **importfunc_kws,
        )
        # > super().__init__(), OR retrieve attributes from instance
        self._get_attributes(
            import_kws=import_kws,
            meta_kws=meta_kws,
            fileimport_kws=fileimport_kws,
        )

        ### Declare types for IDE
        self.data: str | Path | np.ndarray | list[np.ndarray] | Self
        self.imgs: np.ndarray
        self.verbose: str

        ### Total width, height, depth in µm
        # self.x_µm = x_µm
        # self.y_µm = self.imgs.shape[1] * self.x_µm / self.imgs.shape[2]
        # self.pixel_length = self.x_µm / self.imgs.shape[2]
        # self.spacing = (self.x_µm, self.y_µm)
        # ### Metadata for (individual) images
        # self.pixel_length = pixel_length
        # self.unit = unit

        # ### Convert self.imgs into type ImgWithMetadata
        # self._assign_metadata_to_folders()

        # # == Other ==
        # ### Define scalebar length here, required by e.g. mip
        # self.scalebar_length = scalebar_length

    # == Import from parent Instance ===================================
    #
    def _get_attributes(
        self, import_kws: dict, meta_kws: dict, fileimport_kws: dict
    ):
        """Import images from another Imgs instance. This will transfer
        all attributes from the Imgs instance. Methods are transferred
        by inheritance, because we want the option to import images at
        every stage of the processing pipeline."""

        ### Transfer all attributes, if instance is passed
        if isinstance(import_kws["data"], type(self)):
            super().from_instance(
                **import_kws,
                **meta_kws,
            )
        # > If not, call the parent __init__ method
        else:
            super().__init__(
                **import_kws,
                **meta_kws,
                **fileimport_kws,
            )

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

        imgs = scalebar.burn_scalebars(
            imgs=imgs,
            # slice=slice,
            length=self.scalebar_length,
            pixel_length=self.pixel_length,
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
        cmap: str = "gist_ncar",
        max_cols: int = 2,
        scalebar: bool = False,
        scalebar_kws: dict = dict(),
        colorbar=True,
        saveto: str = None,
        **imshow_kws,
    ) -> None:
        """Show the images"""

        ### Make copy to ensure
        _imgs = self.imgs.copy()

        ### Update KWS
        scalebar_KWS = dict(
            length=self.scalebar_length,
            pixel_length=self.pixel_length,
        )
        scalebar_KWS.update(scalebar_kws)

        ### Update kwargs
        _imshow_kws = dict(
            imgs=_imgs,
            max_cols=max_cols,
            cmap=cmap,
            scalebar=scalebar,
            scalebar_kws=scalebar_KWS,
            colorbar=colorbar,
            # saveto=None,
        )
        _imshow_kws.update(imshow_kws)

        ### Total number of images
        T = self._shape_original[0]

        ### MAKE IMAGE
        fig, axes = imageplots.imshow(**_imshow_kws)

        ### Add Ax titles
        for i, ax in enumerate(axes.flat):
            if i >= len(self.imgs):
                break
            # > get correct index if sliced

            _i = self._slice_indices[i] if self._slice else i

            img = _imgs[i]  # > retrieve image
            _ax_tit = (
                f"Image {i+1}/{T} (i={_i}/{T-1})"
                f"    {img.shape[0]}x{img.shape[1]}  {img.dtype}"
            )

            ### Add Image keys
            if not self.imgnames is None:
                fk = self.imgnames[i]
                path, imgk = fk.split(": ")
                _ax_tit = f"'{path}': '{imgk}'\n" + _ax_tit
            ax.set_title(_ax_tit, fontsize="medium")

        ### Fig title
        _fig_tit = f"{self.path_short}\n - {T} Total images"
        if self._slice:
            _fig_tit += f"; Sliced to {len(_imgs)} image(s) (i=[{self._slice}])"
        imageplots.figtitle_to_plot(_fig_tit, fig=fig, axes=axes)

        plt.tight_layout()

        ### Save
        if not saveto is None:
            ut.saveplot(fname=saveto, verbose=self.verbose)

    def mip(self, scalebar=True, **mip_kws) -> np.ndarray | None:
        ### Make copy in case of burning in scalebar
        _imgs = self.imgs.copy()
        if scalebar:
            # > Put scalebar on first image only
            _imgs[0] = self.burn_scalebars()[0]

        mip = imageplots.mip(imgs=_imgs, **mip_kws)
        plt.show()
        return mip

    def plot_histogram(self, bins=75, log=True) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        dataplots.histogram(self.imgs, bins=bins, log=log)

    #
    # !! == End Class ==================================================


if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    pixel_length = 1.5 * 115.4 * 1024
    Z = Imgs(
        data=path,
        fname_extension="txt",
        verbose=True,
        pixel_length=pixel_length,
        imgname_position=1,
    )
    I = 6

    # %%
    Z.metadata


# %%
def _test_import_from_types(Z, I=6):
    # > Import from Path
    kws = dict(
        verbose=True,
        pixel_length=1.5 * 115.4 * 1024,
        fname_extension="txt",
        imgname_position=1,
    )

    Z1 = Imgs(data=path, **kws)
    # Z1[I].imshow()

    # > Import from np.ndarray
    Z2 = Imgs(data=Z.imgs, **kws)
    # Z2[I].imshow()

    # > Import from list of np.ndarrays
    Z3 = Imgs(data=[im for im in Z.imgs], **kws)
    # Z3[I].imshow()

    # > Import from self
    Z4 = Imgs(data=Z, **kws)
    Z5 = Imgs(Z, **kws)
    Z5[I].imshow()


if __name__ == "__main__":
    pass
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
    pass
    # _test_imshow_method(Z)
