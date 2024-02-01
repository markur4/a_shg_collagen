"""A Class for basic funtionalities of images.
- Import
- Pixel scale to µm
- Add Scalebar
- Simple visualizations
- 
"""
# %%

from typing import Self

import copy

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# > Local
import imagep._imgs.import_imgs as import_imgs
import imagep._plottools.scalebar as scaleb
import imagep._plottools.imageplots as imageplots
import imagep._utils.utils as ut


# from imagep.utils.transforms import Transform


# == Class ImgsImport =====================================================
class ImgsImport:
    """Class for handling Imports of raw image data"""

    DEBUG = False

    def __init__(
        self,
        path: str | Path = None,
        array: np.ndarray = None,
        verbose: bool = True,
    ) -> None:
        ### Make sure that either path or array is given
        self.verbose = verbose

        ### Import images from path or array
        if path is not None and array is not None:
            raise ValueError("Either path or array must be given, not both.")
        elif path is not None:
            self.path = Path(path)
            if self.verbose:
                print(f"=> Importing Images from {self.path_short}...")
            self.imgs: np.ndarray = self.import_imgs(self.path)
        elif array is not None:
            self.path = "external numpy array"
            self.imgs = array.astype(import_imgs.DEFAULT_DTYPE)
        else:
            raise ValueError("Either path or array must be given.")
        
        ### Remember if this object has been sliced
        self._slice:bool | str = False
        self._num_imgs: int = self.imgs.shape[0]
        self._slice_indices: list[int] = list(range(self._num_imgs))

    # == Path ====================================================

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

    def __getitem__(self, val: slice) -> Self | "Imgs":
        # > Create a copy of this instance
        _self = copy.deepcopy(self)
        # > Assign the sliced imgs to the new instance
        # indices = ut.indices_from_slice(slice=val, n_imgs=self.imgs.shape[0])
        
        ### Slice while preserving dimension information
        # > Z[0]
        if isinstance(val, int):
            _self.imgs = self.imgs[[val], ...]
            indices = [val]
        # > Z[1:3]
        elif isinstance(val, slice):
            _self.imgs = self.imgs[val, ...]
            indices = range(*val.indices(self._num_imgs))
        # > Z[1,2,5] 
        elif isinstance(val, tuple):
            _self.imgs = self.imgs[list(val), ...]
            indices = val
        # > or Z[[1,2,5]] pick multiple images
        elif isinstance(val, list):
            _self.imgs = self.imgs[val, ...]
            indices = val
            
            
        
        ### Remember how this object was sliced
        _self._slice = str(val)
        _self._slice_indices = indices
        
        return _self

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
            imgs.append(import_imgs.from_txt(txt))

        ### Convert to numpy array
        imgs = np.array(imgs)

        return imgs

    #
    # == I/O ===========================================================

    def save_as_nparray(self, fname: str | Path) -> None:
        """Save the z-stack to a folder"""
        np.save(fname, self.imgs)

    def load_from_nparray(self, fname: str | Path) -> None:
        """Load the z-stack from a folder"""
        self.imgs = np.load(fname)


#
# == Class ImgsGreyscale ===============================================
class ImgsGreyscale:
    """These images are all greyscale with shape (z, y, x)"""

    def __init__(self) -> None:
        pass


#
# == Class ImgsColored =================================================
class ImgsColored:
    """These images are all colored with shape (z, y, x, 3)"""

    def __init__(self) -> None:
        pass


#
# == Class ImgsSameSize ================================================
class ImgsSameSize(ImgsImport):
    """These images all have the same Size with"""

    def __init__(
        self,
        path: str | Path = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(path, verbose)

        ### Check if all images have the same size
        self._check_size()

    def _check_size(self) -> None:
        """Check if all images have the same size"""
        sizes = [img.shape for img in self.imgs]
        if not all([size == sizes[0] for size in sizes]):
            raise ValueError(
                f"Not all images have the same size. Found these {set(sizes)}!"
            )


# %%
# == Class Imgs ========================================================


class Imgs(ImgsImport):
    """Interface for handling image types."""

    def __init__(
        self,
        path: str | Path = None,
        verbose: bool = True,
        x_µm: float = 200.0,
        scalebar_microns: int = 10,
    ):
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
        # todo: Add a class for images of different sizes
        # todo: Make a colored image class

        ### Import Images
        super().__init__(path=path, verbose=verbose)

        ### Total width, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.imgs.shape[1] * self.x_µm / self.imgs.shape[2]
        self.pixel_size = self.x_µm / self.imgs.shape[2]
        self.spacing = (self.x_µm, self.y_µm)

        # == Other ==
        ### Define scalebar length here, required by e.g. mip
        self.scalebar_microns = scalebar_microns

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
            #> get correct index if sliced

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
            tit += f"; {len(_imgs)} sliced images: [{self._slice}]"
        
        ### get number of rows in axes
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
    Z = Imgs(path=path, verbose=True, x_µm=1.5 * 115.4)
    I = 6


# %%
def _test_imshow_method(Z):
    kws = dict(
        max_cols=2,
        scalebar=True,
    )

    Z[0].imshow(**kws)
    Z[0,3,5,19].imshow(**kws)
    Z[1:3].imshow(**kws)
    Z[0:10:3].imshow(**kws)
    Z[0:10:2].imshow(**kws)
    Z[[1,3,6,7,8,9,15]].imshow(**kws)
    plt.show()  # >   show last one

    ### Check if scalebar is not burned it
    plt.imshow(Z.imgs[0])
    plt.suptitle("no scalebar should be here")
    plt.show()


if __name__ == "__main__":
    _test_imshow_method(Z)
