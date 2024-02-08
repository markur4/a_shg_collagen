"""Plotting functions that display images"""

# %%
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

# > local imports
import imagep._utils.utils as ut
import imagep._plots.scalebar as scaleb


# from imagep.images.imgs import Imgs

# %%
# == Testdata ==========================================================
if __name__ == "__main__":
    # !! Import must not be global, or circular import
    from imagep.images.imgs import Imgs

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Imgs(data=path, verbose=True, x_µm=1.5 * 115.4)
    print("pixelsize=", Z.pixel_length)
    I = 6


# %%
# == UTILS =============================================================
# def _fname(fname: bool | str, extension=".png") -> Path:
#     """Tool to get filename for saving"""
#     if not isinstance(fname, str):
#         raise ValueError("You must provide a filename for save")
#     fname = Path(fname).with_suffix(".png")
#     return fname


def figtitle_to_plot(title: str, fig: plt.Figure, axes: np.ndarray) -> None:
    """Makes a suptitle for figures"""
    bbox_y = 1.05 if axes.shape[0] <= 2 else 1.01
    fig.suptitle(title, ha="left", x=0.01, y=bbox_y, fontsize="large")


# %%
# == colorbar ==========================================================
def _colorbar_to_ax(
    plt_img: mpl.image,
    ax: plt.Axes,
    percentiles: tuple = None,
    minmax: tuple = None,
) -> plt.colorbar:
    """Add colorbar to axes"""
    # > Colorbar

    cb = plt.colorbar(
        mappable=plt_img,
        ax=ax,
        fraction=0.04,  # > Size colorbar relative to ax
        orientation="vertical",
    )
    # > plot metrics onto colorbar
    kws = dict(
        xmin=0,
        xmax=15,
        zorder=100,
    )

    ### Plot percentiles
    if percentiles is not None:
        perc50, perc75, perc99 = percentiles
        ### min max lines
        line_max = cb.ax.hlines(
            minmax[1],
            ls="solid",
            lw=2,
            label=f"max",
            colors="red",
            **kws,
        )
        line_99 = cb.ax.hlines(
            perc99,
            ls=(0, [0.4, 0.6]),  # > (offset, (on, off))
            lw=3,
            label=f"99th",
            colors="white",
            **kws,
        )
        line_75 = cb.ax.hlines(
            perc75,
            ls=(0.5, [1.5, 0.5]),
            lw=2,
            label=f"75th",
            colors="white",
            **kws,
        )
        line_50 = cb.ax.hlines(
            perc50,
            ls="solid",
            lw=1.5,
            label=f"median",
            colors="white",
            **kws,
        )
        line_min = cb.ax.hlines(
            minmax[0],
            ls="solid",
            lw=2,
            label=f"min",
            colors="red",
            **kws,
        )
        for line in [line_99, line_75, line_50]:
            line.set_gapcolor("black")

        ### Add as minor ticks
        cb.ax.set_yticks([minmax[0], minmax[1], perc50, perc75, perc99])
        # cb.ax.tick_params(
        #     which="minor",
        #     labelsize="small",
        # )

    ### Format ticklabels
    form = mpl.ticker.FuncFormatter(ut.format_num)
    cb.ax.yaxis.set_minor_formatter(form)
    cb.ax.yaxis.set_major_formatter(form)

    return cb


# %%
# == imshow ============================================================
def imshow(
    imgs: np.ndarray,
    cmap: str = "gist_ncar",
    max_cols: int = 2,
    scalebar: bool = False,
    scalebar_kws: dict = dict(),
    colorbar=True,
    share_cmap: bool = True,
    saveto: str = None,
    **imshow_kws,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Show the images"""

    ### If single image, make it nested
    if len(imgs.shape) == 2:
        imgs = np.array([imgs])

    ### Always make copy when showing
    # !! Keep raw, since scalebar changes metrics
    _imgs = imgs.copy()
    _imgs_raw = imgs.copy()

    ### if single image, make it nested
    if len(_imgs.shape) == 2:
        _imgs = np.array([_imgs])

    ### Scalebar
    if scalebar:
        ### Default values for scalebar
        ut.check_arguments(
            scalebar_kws,
            kws_name="scalebar_kws",
            required=["pixel_length"],
        )
        _imgs = scaleb.burn_scalebars(
            imgs=_imgs,
            **scalebar_kws,
        )

    ### Update kwargs
    imshow_KWS = dict(cmap=cmap)
    imshow_KWS.update(imshow_kws)

    ### Number of rows and columns
    # > Columns is 1, but maximum max_cols
    # > Fill rows with rest of images
    n_cols = 1 if len(_imgs) == 1 else max_cols
    n_rows = int(np.ceil(len(_imgs) / n_cols))

    ### Plot
    fig, axes = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        figsize=(n_cols * 5, n_rows * 5),
        squeeze=False,
        # sharey=True,
    )

    # > Calculate limits for pixel and colorbar
    min_all, max_all = np.min(_imgs), np.max(_imgs)

    ### Fillaxes
    for i, ax in enumerate(axes.flat):
        if i >= len(_imgs):
            ax.axis("off")
            continue
        # > Retrieve image
        img: np.ndarray = _imgs[i]

        # > PLOT IMAGE
        plt_img = ax.imshow(img, **imshow_KWS)
        ax.axis("off")

        # > All images have same limits
        if share_cmap:
            plt_img.set_clim(min_all, max_all)

        # > Colorbar
        if colorbar:
            #!! Calculate from raw images, since scalebar changes pixel values
            p = np.percentile(_imgs_raw[i], [50, 75, 99])
            minmax = np.min(_imgs_raw[i]), np.max(_imgs_raw[i])
            cb = _colorbar_to_ax(
                plt_img=plt_img, percentiles=p, ax=ax, minmax=minmax
            )

        ### axes title
        AXTITLE = (
            f"Image {i+1}/{len(_imgs)},"  # > don't provide index
            f"  {img.shape[0]}x{img.shape[1]}  {img.dtype}"
        )
        ax.set_title(f"Image {i+1}/{len(_imgs)}", fontsize="medium")

    ### legend for colorbar lines
    handles, labels = cb.ax.get_legend_handles_labels()

    bbox_x = 1.04 if n_cols == 1 else 1.02
    bbox_y = 1.12 if n_rows == 1 else 1.06
    bbox_y = 1.06 if n_rows == 2 else bbox_y
    bbox_y = 1.04 if n_rows >= 3 else bbox_y
    legend = fig.legend(
        title="Percentiles",
        loc="upper right",
        bbox_to_anchor=(bbox_x, bbox_y),  # > (x, y)
        handles=handles,
        labels=labels,
        fontsize=10,
        framealpha=0.2,
    )
    frame = legend.get_frame()
    frame.set_facecolor("black")

    ### aligned left
    FIGTITLE = f"{imgs.shape[0]} images, {imgs.dtype}"
    figtitle_to_plot(FIGTITLE, fig=fig, axes=axes)

    # plt.suptitle(f"{imgs.shape[0]} images, {imgs.dtype}", ha="left", x=0.05)
    plt.tight_layout()

    if saveto:
        ut.saveplot(fname=saveto, verbose=True)

    return fig, axes


def _test_imshow_global(Z):
    kws = dict(
        max_cols=2,
        scalebar=True,
        scalebar_kws=dict(
            pixel_length=0.05,  # !! must be provided when scalebar
        ),
    )
    imgs = Z.imgs

    imshow(imgs[0], **kws)
    imshow(imgs[[0, 1, 2]], **kws)
    imshow(imgs[0:10:2], **kws)

    ### Check if scalebar is not burned it
    plt.imshow(Z.imgs[0])
    plt.suptitle("no scalebar should be here")
    plt.show()


if __name__ == "__main__":
    pass
    # _test_imshow_global(Z)


# %%
# == mip ===============================================================
def mip(
    imgs: np.ndarray,
    axis: int = 0,
    show=True,
    return_array=False,
    fname: bool | str = False,
    colormap: str = "gist_ncar",
) -> np.ndarray | None:
    """Maximum intensity projection across certain axis"""

    mip = imgs.max(axis=axis)

    if show:
        plt.imshow(
            mip,
            cmap=colormap,
            interpolation="none",
        )
        # plt.show()

    if fname:
        fname = _fname(fname)
        plt.imsave(fname=fname, arr=mip, cmap=colormap)

    if return_array:
        return mip
