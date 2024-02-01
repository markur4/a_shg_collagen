"""Plotting functions that display images"""
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

# > local imports
import imagep._utils.utils as ut
import imagep._plottools.scalebar as scaleb


# %%
# == Testdata ==========================================================
if __name__ == "__main__":
    # !! Import must not be global, or circular import
    from imagep._imgs.imgs import Imgs

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Imgs(path=path, verbose=True, x_µm=1.5 * 115.4)
    print("pixelsize=", Z.pixel_size)
    I = 6


# %%
# == UTILS =============================================================
def _fname(fname: bool | str, extension=".png") -> Path:
    """Tool to get filename for saving"""
    if not isinstance(fname, str):
        raise ValueError("You must provide a filename for save")
    fname = Path(fname).with_suffix(".png")
    return fname


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
    fname: bool | str = False,
    **imshow_kws,
) -> tuple[plt.Figure, plt.Axes]:
    """Show the images"""
    ### Always make copy when showing
    _imgs = imgs.copy()

    ### if single image, make it nested
    if len(_imgs.shape) == 2:
        _imgs = np.array([_imgs])

    ### Scalebar
    # !! Done before retrieving images, so
    if scalebar:
        ### Default values for scalebar
        ut.check_arguments(
            scalebar_kws,
            kws_name="scalebar_kws",
            required=["pixel_size"],
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
        plt_img.set_clim(min_all, max_all)

        # > Colorbar
        if colorbar:
            #!! Calculate from raw images, since adding scalebar changes pixel values
            p = np.percentile(_imgs[i], [50, 75, 99])
            minmax = np.min(_imgs[i]), np.max(_imgs[i])
            cb = _colorbar_to_ax(
                plt_img=plt_img, percentiles=p, ax=ax, minmax=minmax
            )

    ### legend for colorbar lines

    handles, labels = cb.ax.get_legend_handles_labels()

    bbox_x = 1.04 if n_cols == 1 else 1.02
    bbox_y = 1.12 if n_rows == 1 else 1.06
    bbox_y = 1.06 if n_rows == 2 else bbox_y
    bbox_y = 1.04 if n_rows >= 3 else bbox_y
    fig.legend(
        title="Percentiles",
        loc="upper right",
        bbox_to_anchor=(bbox_x, bbox_y),  # > (x, y)
        handles=handles,
        labels=labels,
        fontsize=10,
        framealpha=0.2,
    )

    plt.tight_layout()

    return fig, axes


# def _test_imshow_global(Z):
#     kws = dict(
#         max_cols=2,
#         scalebar_kws=dict(
#             pixel_size=0.05,  # !! must be provided when scalebar
#         ),
#     )
#     imgs = Z.imgs

#     imshow(imgs, slice=0, **kws)
#     imshow(imgs, slice=[1, 2], **kws)
#     imshow(imgs, slice=[1, 2, 3], **kws)
#     imshow(imgs, slice=(1, 10, 2), **kws)  # > start stop step

#     ### Check if scalebar is not burned it
#     plt.imshow(Z[0])
#     plt.suptitle("no scalebar should be here")
#     plt.show()


# if __name__ == "__main__":
#     _test_imshow_global(Z)


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
