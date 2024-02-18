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
import imagep._configs.rc as rc

# import imagep.images.importtools as importtools
# from imagep.images.imgs_import import CollectionImport
import imagep._utils.types as T
from imagep.images.mdarray import mdarray
from imagep.images.stack_meta import StackMeta
import imagep._plots.scalebar as scaleb
import imagep._plots.imageplots as imageplots
import imagep._plots.dataplots as dataplots

if TYPE_CHECKING:
    from imagep.images.stack import Stack
# from imagep.utils.transforms import Transform


# %%
# ======================================================================
# == Class Stack =======================================================
class Stack(StackMeta):
    """Interface for handling images in a stack."""

    def __init__(
        self,
        ### CollectionImport kws:
        data: T.source_of_imgs = None,
        verbose: bool = True,
        ### CollectionMeta kws:
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
        """Interface for image-stack functionalities.
        - Block super()__init__ if to avoid re-loading images
        - Show content of collection (`imshow()`)


        :param data: pathlike string or object
        :type data: str | Path | np.ndarray | list[np.ndarray] | Self
        :param x_µm: Total width of image in µm, defaults to 200
        :type x_µm: float, optional
        :param scalebar_length: Length of scalebar in µm. This won't
            add a scalebar, it's called once needed, defaults to
            10
        :type scalebar_length: int, optional
        """

        ### Collect kws
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
        )
        ### super().__init__(), OR retrieve attributes from instance
        self._get_attributes(
            import_kws=import_kws,
            meta_kws=meta_kws,
            fileimport_kws=fileimport_kws,
            **importfunc_kws,
        )

        ### Declare types for IDE
        self.data: str | Path | np.ndarray | list[np.ndarray] | Self
        self.imgs: np.ndarray
        self.verbose: str

    # == Import from parent Instance ===================================
    #
    def _get_attributes(
        self,
        import_kws: dict,
        meta_kws: dict,
        fileimport_kws: dict,
        **importfunc_kws,
    ):
        """Import images from another Collection instance. This will transfer
        all attributes from the Collection instance. Methods are transferred
        by inheritance, because we want the option to import images at
        every stage of the processing pipeline."""

        ### Get user input on source data
        data = import_kws["data"]

        ### If data is an instance of Collection, just transfer its attributes
        if isinstance(data, type(self)):
            # todo: User modify metadata when re-importing from instance
            super().from_instance(
                **import_kws,
                # **meta_kws, # !! Assume user wants same metadata
            )
        ### If data is not a Collection, run full inheritance
        else:
            super().__init__(
                **import_kws,
                **meta_kws,
                **fileimport_kws,
                **importfunc_kws,
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

        imgs = scaleb.burn_scalebars(
            imgs=imgs,
            length=self.scalebar_length,
            # pixel_length=self.pixel_length, #!! provide by img metadata
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
        colorbar:bool=True,
        share_cmap: bool = False,
        batch_size: int = None,
        save_as: str = None,
        ret: bool = False,
        **imshow_kws,
    ) -> None:
        """Show the images"""

        ### Make copy to ensure
        _imgs = self.imgs.copy()

        ### Update KWS
        scalebar_KWS = dict(
            length=self.scalebar_length,
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
        )
        _imshow_kws.update(imshow_kws)

        ### Total number of images
        T = self._shape_original[0]

        ### MAKE IMAGE
        if batch_size is None:
            fig, axes = imageplots.imshow(**_imshow_kws)
            fig_axes = (fig, axes)
        else:
            fig_axes = imageplots.plot_images_in_batches(
                batch_size=batch_size, **_imshow_kws
            )
        
        for fig, axes in fig_axes:
            ### Add Ax titles ==========================================
            for i, ax in enumerate(axes.flat):
                if i >= len(self.imgs):
                    break
                # > get correct index if sliced
                _i_tot = self._slice_indices[i] if self._sliced else i
                # > retrieve image
                img: mdarray = _imgs[i]
                # > ax title
                imageplots.axtitle_to_plot(ax, img, i=i, i_tot=_i_tot, T=T)

            ### Fig title
            _fig_tit = f"{self.paths_pretty}\n - {T} Total images"
            if self._sliced:
                _fig_tit += (
                    f"; Sliced to {len(_imgs)} image(s) (i=[{self._sliced}])"
                )
            imageplots.figtitle_to_plot(_fig_tit, fig=fig, axes=axes)

            plt.tight_layout()

        ### Save
        if not save_as is None:
            imageplots.savefig(save_as=save_as, verbose=self.verbose)

        ### Return or show
        if ret:
            return fig, axes
        else:
            plt.show()

        # !!

    def mip(self, scalebar=True, **mip_kws) -> np.ndarray | None:
        ### Make copy in case of burning in scalebar
        _imgs = self.imgs.copy()
        if scalebar:
            # > Put scalebar on first image only
            _imgs[0] = self.burn_scalebars()[0]

        mip = imageplots.plot_mip(imgs=_imgs, **mip_kws)
        plt.show()
        return mip

    def plot_histogram(self, bins=75, log=True) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        dataplots.histogram(self.imgs, bins=bins, log=log)

    #
    # == Utils =========================================================

    def copy(self) -> Self:
        """Makes a full copy of itself"""
        return copy.deepcopy(self)

    #
    # !! == End Class ==================================================


if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    pixel_length = (1.5 * 115.4) / 1024
    Z = Stack(
        data=path,
        fname_extension="txt",
        verbose=True,
        pixel_length=pixel_length,
        imgname_position=1,
        sort=True,
        invertorder=True,
    )
    I = 6
    # %%
    print(type(Z.imgs))
    print(Z.imgs.shape)
    print(Z.imgs[0])
    print(type(Z.imgs[0]))

    # %%
    Z.mip()

    # %%
    Z.mip(axis=1)

    # %%
    Z[0:20:3].imshow()


# %%
def _test_import_from_types(Z, I=6):
    # > Import from Path
    kws = dict(
        verbose=True,
        pixel_length=[1.5 * 115.4 * 1024],
        fname_extension="txt",
        imgname_position=1,
        sort=True,
        invertorder=True,
    )

    print("IMPORT FROM PATH:")
    Z1 = Stack(data=path, **kws)
    Z1[I].imshow()

    print("IMPORT FROM ARRAY:")
    Z2 = Stack(data=Z.imgs, **kws)
    Z2[I].imshow()

    print("IMPORT FROM LIST OF ARRAYS:")
    # > Import from list of np.ndarrays
    Z3 = Stack(data=[im for im in Z.imgs], **kws)
    Z3[I].imshow()

    print("IMPORT FROM SELF:")
    Z4 = Stack(data=Z, **kws)
    Z4[I].imshow()
    pprint(Z4.metadata, compact=True)

    print("IMPORT FROM SELF (as *arg):")
    Z5 = Stack(Z, **kws)
    Z5[I].imshow()
    pprint(Z5.metadata, compact=True)


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
    _test_imshow_method(Z)
