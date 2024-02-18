"""Plotting functions that display images"""

# %%
from typing import TYPE_CHECKING
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import Path

# > local imports
import imagep._utils.utils as ut
import imagep._utils.types as T
import imagep._plots.scalebar as scaleb


# from imagep.images.imgs import Imgs

# %%
# == Testdata ==========================================================
if __name__ == "__main__":
    # !! Import must not be global, or circular import
    from imagep.images.stack import Stack

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Stack(
        data=path,
        # fname_pattern="*.txt",
        fname_extension="txt",
        imgname_position=-1,
        verbose=True,
        pixel_length=(1.5 * 115.4) / 1024,
    )
    I = 6
    Z


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


def axtitle_to_plot(
    ax: plt.Axes,
    img: T.array,
    i: int,
    i_tot: int,
    T: int,
) -> None:
    """Adds title to axes"""
    _ax_tit = (
        f"Image {i+1}/{T} (i={i_tot}/{T-1})"
        f"    {img.shape[0]}x{img.shape[1]}  {img.dtype}"
    )
    # > Add Image keys
    if img.name != "unnamed":
        folder = Path(img.folder).parent
        _ax_tit = f"'{folder}': '{img.name}'\n" + _ax_tit
    ax.set_title(_ax_tit, fontsize="medium")


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
        alpha=1,  # !! override
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


# == refactored ======================================================
def _plot_image(
    ax: plt.Axes,
    img: T.array,
    imshow_kws: dict,
    share_cmap: bool,
    min_max: tuple,
    colorbar: bool,
    _imgs_raw: T.array,
    i,
):
    plt_img = ax.imshow(img, **imshow_kws)
    ax.axis("off")

    if share_cmap:
        plt_img.set_clim(*min_max)

    if colorbar:
        p = np.percentile(_imgs_raw[i], [50, 75, 99])
        minmax = np.min(_imgs_raw[i]), np.max(_imgs_raw[i])
        cb = _colorbar_to_ax(
            plt_img=plt_img, percentiles=p, ax=ax, minmax=minmax
        )

    if hasattr(img, "name"):
        axtitle_to_plot(
            ax=ax,
            img=img,
            i=i,
            i_tot=len(_imgs_raw) - 1,
            T=len(_imgs_raw),
        )
    else:
        ax.set_title(f"Image {i+1}/{len(_imgs_raw)}", fontsize="medium")

    return cb


def _plot_legend(fig, cb, n_cols, n_rows):
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


def imshow(
    imgs: np.ndarray,
    cmap: str = "gist_ncar",
    max_cols: int = 2,
    scalebar: bool = False,
    scalebar_kws: dict = dict(),
    colorbar=True,
    share_cmap: bool = True,
    min_max: tuple = None,
    save_as: str = None,
    ret=True,
    **imshow_kws,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Show the images"""

    if len(imgs.shape) == 2:
        imgs = np.array([imgs])

    _imgs = imgs.copy()
    _imgs_raw = imgs.copy()

    if len(_imgs.shape) == 2:
        _imgs = np.array([_imgs])

    if scalebar:
        _imgs = scaleb.burn_scalebars(
            imgs=_imgs,
            **scalebar_kws,
        )

    imshow_KWS = dict(cmap=cmap)
    imshow_KWS.update(imshow_kws)

    n_cols = 1 if len(_imgs) == 1 else max_cols
    n_rows = int(np.ceil(len(_imgs) / n_cols))

    fig, axes = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        figsize=(n_cols * 5, n_rows * 5),
        squeeze=False,
    )

    if min_max is None:
        min_max = (np.min(_imgs), np.max(_imgs))

    for i, ax in enumerate(axes.flat):
        if i >= len(_imgs):
            ax.axis("off")
            continue

        img: np.ndarray = _imgs[i]

        cb = _plot_image(
            ax,
            img,
            imshow_KWS,
            share_cmap,
            min_max,
            colorbar,
            _imgs_raw,
            i,
        )

    _plot_legend(fig, cb, n_cols, n_rows)

    FIGTITLE = f"{imgs.shape[0]} images, {imgs.dtype}"
    figtitle_to_plot(FIGTITLE, fig=fig, axes=axes)

    plt.tight_layout()

    if save_as:
        savefig(save_as=save_as, verbose=True)

    if ret:
        return fig, axes
    else:
        plt.show()


def _test_imshow_global(imgs: np.ndarray):
    kws = dict(
        max_cols=2,
        scalebar=True,
        scalebar_kws=dict(
            pixel_length=0.05,  # !! must be provided when scalebar
        ),
        ret=False,
    )
    # imgs = Z.imgs

    imshow(imgs[0], **kws)
    imshow(imgs[[0, 1, 2]], **kws)
    imshow(imgs[0:10:2], **kws)

    ### Check if scalebar is not burned it
    plt.imshow(imgs[0])
    plt.suptitle("no scalebar should be here")
    plt.show()


if __name__ == "__main__":
    pass
    _test_imshow_global(imgs = Z.imgs)


# %%
# == Savefig ===========================================================
def _finalize_fname(fname: str) -> Path:
    """Add .pdf as suffix if no suffix is present"""
    if "." in fname:
        return Path(fname)
    else:
        return Path(fname).with_suffix(".pdf")


def savefig(
    save_as: str,
    verbose: bool = True,
) -> None:
    """Saves current plot to file"""

    if not isinstance(save_as, str):
        raise ValueError(
            f"You must provide a filename for save. Got: '{save_as}'"
        )
    ### Add .pdf as suffix if no suffix is present
    save_as = _finalize_fname(save_as)

    plt.savefig(save_as, bbox_inches="tight")

    if verbose:
        print(f"Saved plot to: {save_as.resolve()}")


# %%
# == Batch Plotting ====================================================
def save_figs_to_pdf(figs_axes, save_as):
    ### Add .pdf as suffix if no suffix is present
    save_as = _finalize_fname(save_as)
    with PdfPages(save_as) as pdf:
        for fig, _ in figs_axes:
            pdf.savefig(fig, bbox_inches="tight")


def plot_images_in_batches(
    imgs,
    batch_size=4,
    save_as: str = None,
    **kwargs,
):
    ### Get range, sice we need that for shared colorbars
    min_max = (np.min(imgs), np.max(imgs))
    
    ### Plot in batches
    figs_axes = []
    for i in range(0, len(imgs), batch_size):
        fig, axes = imshow(
            imgs[i : i + batch_size],
            min_max=min_max,
            **kwargs,
            ret=True,
        )
        figs_axes.append((fig, axes))

    if not save_as is None:
        save_figs_to_pdf(figs_axes, save_as)

    return figs_axes


def _test_batch_plot():
    figs_axes = plot_images_in_batches(
        Z.imgs,
        batch_size=4,
        save_as="test",
    )
    # save_figs_to_pdf(figs_axes, "test.pdf")


if __name__ == "__main__":
    pass
    _test_batch_plot()


# %%
# == mip ===============================================================
def plot_mip(
    imgs: np.ndarray,
    axis: int = 0,
    colormap: str = "gist_ncar",
    ret=False,
    save_as: str = None,
    verbose=True,
) -> np.ndarray | None:
    """Maximum intensity projection across certain axis"""

    mip = imgs.max(axis=axis)

    plt.imshow(
        mip,
        cmap=colormap,
        interpolation="none",
    )

    if not save_as is None:
        savefig(save_as=save_as, verbose=verbose)

    if ret:
        return mip
    else:
        plt.show()
