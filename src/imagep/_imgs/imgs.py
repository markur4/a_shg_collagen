"""A Class for basic funtionalities of images.
- Import
- Pixel scale to µm
- Add Scalebar
- Simple visualizations
- 
"""
# %%
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# > Local
import imagep._imgs.import_imgs as import_imgs
import imagep._plottools.scalebar as scalebar
import imagep._plottools.imageplots
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

    def burn_scalebar(
        self,
        imgs: np.ndarray = None,
        slice: str | int | list[int] | tuple[int] = "all",
        xy_pad: tuple[float] = (0.05, 0.05),
        thickness_px: int = 20,
        text_color: tuple[int] | str = None,
    ) -> np.ndarray:
        """Burns scalebar to images in stack. By default, adds only to
        the first image, but can be changed with indexes.
        :param imgs: Images to burn scalebar to, defaults to None
        :type imgs: np.ndarray, optional
        :param img_slice: Part of images to burn scalebar to. Options
        are `"all"`, `int`, `list[int]` or `tuple(start,stop,step)`,
        defaults to "all"
        :type img_slice: str | int | tuple[int], optional
        :param xy_pad: Distance of scalebar from bottom left corner in
        percent of image width and height, defaults to (0.05, 0.05)
        :type xy_pad: tuple[float], optional
        :param thickness: Thickness of scalebar in pixels, defaults to
        3
        :type thickness: int, optional
        :param text_color: Color of text, defaults to "white"
        :type text_color: str, optional
        """
        ## Take imgs from self if not given
        imgs = self.imgs if imgs is None else imgs

        ### If all, burn to all images
        indices = ut.indices_from_slice(slice=slice, n_imgs=imgs.shape[0])

        for i in indices:
            imgs[i] = scalebar.burn_scalebar_to_img(
                img=imgs[i],
                microns=self.scalebar_microns,
                pixel_size=self.pixel_size,
                thickness_px=thickness_px,
                xy_pad=xy_pad,
                bar_color=imgs.max(),
                frame_color=imgs.max() * 0.9,
            )
            imgs[i] = scalebar.burn_micronlength_to_img(
                img=imgs[i],
                microns=self.scalebar_microns,
                thickness_px=thickness_px,
                xy_pad=xy_pad,
                color=text_color,
            )
        return imgs

    #
    # == Plots =========================================================

    def imshow(
        self,
        imgs: np.ndarray = None,
        slice: str | int | list[int] | tuple[int] = 0,
        cmap: str = "gist_ncar",
        max_cols: int = 1,
        scalebar=True,
        colorbar=True,
        **imshow_kws,
    ) -> None:
        """Show the images"""

        ###
        form = lambda num: ut.format_num(num, exponent=ut._EXPONENT)

        ### Take imgs from self if not given
        imgs = self.imgs.copy() if imgs is None else imgs

        ### Scalebar
        if scalebar:
            imgs = self.burn_scalebar(imgs=imgs, slice=slice)

        ### Retrieve images
        indices = ut.indices_from_slice(slice=slice, n_imgs=imgs.shape[0])
        imgs = imgs[indices]

        ### Update kwargs
        KWS = dict(
            cmap=cmap,
        )
        KWS.update(imshow_kws)

        ### Number of rows and columns
        # > Columns is 1, but maximum max_cols
        # > Fill rows with rest of images
        n_cols = 1 if len(indices) == 1 else max_cols
        n_rows = int(np.ceil(len(indices) / n_cols))

        ### Plot
        ### Temporarily set font color to white and background to black
        fig, axes = plt.subplots(
            ncols=n_cols,
            nrows=n_rows,
            figsize=(n_cols * 5, n_rows * 5),
            squeeze=False,
        )
        ### Fillaxes
        for i, ax in enumerate(axes.flat):
            if i >= len(indices):
                ax.axis("off")
                continue
            # > Retrieve image
            img: np.ndarray = imgs[i]

            # > PLOT IMAGE
            _im = ax.imshow(img, **KWS)
            ax.axis("off")

            # > Ax title
            axtit = (
                f"#Image {indices[i]+1}/{len(self.imgs)} (i = {indices[i]})"
                f"    {img.shape[0]}x{img.shape[1]}  {img.dtype}"
                # f"\nmin={form(img.min())}  mean={form(img.mean())}  max={form(img.max())}"
            )
            ax.set_title(axtit, fontsize=10)

            # > Colorbar
            if colorbar:
                cb = plt.colorbar(
                    mappable=_im,
                    ax=ax,
                    fraction=0.04,  # > Size colorbar relative to ax
                )
                # > plot metrics onto colorbar
                hl_kws = dict(xmin=0, xmax=10)
                perc99 = np.percentile(img, 99)
                perc75 = np.percentile(img, 75)
                mean = img.mean()
                cb.ax.hlines(
                    perc99,
                    label=f"99th percentile",
                    colors="black",
                    **hl_kws,
                )
                cb.ax.hlines(
                    perc75,
                    label=f"75th percentile",
                    colors="grey",
                    **hl_kws,
                )
                cb.ax.hlines(
                    mean, label=f"mean", colors="white", **hl_kws
                )
                
                # > add extra ticks
                # ticks = list(cb.ax.get_yticks())
                # cb.ax.set_yticks(ticks + [perc99, perc75, mean])

        ### legend for colorbar lines
        handles, labels = cb.ax.get_legend_handles_labels()
        fig.legend(
            # title="Colorbar",
            loc="upper right",
            bbox_to_anchor=(1, 1),
            handles=handles,
            labels=labels,
            fontsize=10,
            fancybox=False,
            framealpha=0.2,
        )

        # ### Background color
        fig.patch.set_facecolor("darkgrey")

        ### Fig title
        tit = f"{self.path_short}\n{len(self.imgs)} images"
        fig.suptitle(tit, ha="left", x=0.01, y=1.00, fontsize=12)

        plt.tight_layout()
        plt.show()

    def mip(self, scalebar=True, **mip_kws) -> np.ndarray | None:
        ### Make copy in case of burning in scalebar
        imgs = self.imgs.copy()
        if scalebar:
            # > Put scalebar on first image only
            imgs = self.burn_scalebar(imgs=imgs, slice=0)
            
        mip = imagep._plottools.imageplots.mip(imgs=imgs, **mip_kws)
        plt.show()
        return mip

    @staticmethod
    def plot_brightness_distribution(
        imgs: np.ndarray, bins=75, log=True
    ) -> None:
        """Plot the brightness distribution of the z-stack as
        histogram"""

        plt.hist(imgs.flatten(), bins=bins, log=log)
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
def _test_imshow(Z):
    kws = dict(
        max_cols=2,
        scalebar=True,
    )

    Z.imshow(slice=0, **kws)
    Z.imshow(slice=[1, 2], **kws)
    Z.imshow(slice=[1, 2, 3], **kws)
    Z.imshow(slice=(1, 10, 2), **kws)  # > start stop step

    ### Check if scalebar is not burned it
    plt.imshow(Z[0])
    plt.show()


if __name__ == "__main__":
    _test_imshow(Z)
