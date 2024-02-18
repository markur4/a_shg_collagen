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
from imagep.images.l2Darrays import l2Darrays


# from imagep.images.imgs import Imgs

# %%
# :: Testdata ==========================================================
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
# == img to ax ==========================================================
def _colorbar_to_ax(
    ax_img: mpl.image,
    ax: plt.Axes,
    percentiles: tuple = None,
    min_max: tuple = None,
) -> plt.colorbar:
    """Add colorbar to axes"""
    # > Colorbar

    cb = plt.colorbar(
        mappable=ax_img,
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
            min_max[1],
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
            min_max[0],
            ls="solid",
            lw=2,
            label=f"min",
            colors="red",
            **kws,
        )
        for line in [line_99, line_75, line_50]:
            line.set_gapcolor("black")

        ### Add as minor ticks
        cb.ax.set_yticks([min_max[0], min_max[1], perc50, perc75, perc99])
        # cb.ax.tick_params(
        #     which="minor",
        #     labelsize="small",
        # )

    ### Format ticklabels
    form = mpl.ticker.FuncFormatter(ut.format_num)
    cb.ax.yaxis.set_minor_formatter(form)
    cb.ax.yaxis.set_major_formatter(form)

    return cb


def _plot_img_to_ax(
    ### Select image
    img: T.array,
    ax: plt.Axes,
    i_in_ax: int,
    ### kws
    colorbar: bool,
    share_cmap: bool,
    ### Info persistant across batches of imgs
    imgs_raw: T.array,
    i_in_batch: int,
    ### kws for ax.imshow()
    **ax_imshow_kws,
) -> tuple[plt.Axes]:
    ### Extract parameters from imgs_raw
    min_max = (np.min(imgs_raw), np.max(imgs_raw))
    i_total = len(imgs_raw) - 1
    i_in_total = i_in_batch + i_in_ax
    img_raw = imgs_raw[i_in_total]

    ### Plot img to ax
    ax_img = ax.imshow(img, **ax_imshow_kws)

    ### Remove x- and y-axis
    ax.axis("off")

    ### Define color limits by image stack
    if share_cmap:
        ax_img.set_clim(*min_max)

    ### Colorbar
    if colorbar:
        p = np.percentile(img_raw, [50, 75, 99])
        min_max_img = np.min(img_raw[i_in_total]), np.max(img_raw[i_in_total])
        cb = _colorbar_to_ax(
            ax=ax,
            ax_img=ax_img,
            percentiles=p,
            min_max=min_max_img,
        )

    ### Ax title
    axtitle = _axtitle_from_img(
        img=img_raw,
        i_in_total=i_in_total,
        i_total=i_total,
    )
    ax.set_title(axtitle, fontsize="medium")

    return ax, cb


def _axtitle_from_img(
    img: T.array,
    i_in_total: int,
    i_total: int,
) -> None:
    """Adds title to axes"""
    
    l = []
    
    ### Name
    if hasattr(img, "name"):
        # if img.name != "unnamed":
        folder = Path(img.folder).parent
        l.append(f"'{folder}': '{img.name}'")
    
    ### Index, shape, dtype
    l.append(
        f"Image {i_in_total+1}/{i_total+1} (i={i_in_total}/{i_total})"
        f"    {img.shape[0]}x{img.shape[1]}  {img.dtype}"
    )
    return "\n".join(l)


# %%
# == imshow ============================================================
def _get_folders_from_imgs(imgs: l2Darrays) -> list[str]:
    """Extracts folder names from imgs"""
    folders = set()
    for img in imgs:
        if hasattr(img, "folder"):
            folders.add(img.folder)
    return list(folders)

def _figtitle_from_imgs(imgs: l2Darrays | T.array) -> str:
    """Makes a suptitle for figures"""
    
    ### Extract parameters from imgs
    if isinstance(imgs, l2Darrays):
        types = imgs.dtypes_pretty
        shapes = imgs.shapes
    else:
        types = str(imgs.dtype)
        shapes = imgs.shape
    
    l = []
    ### Length, types
    l.append(f"{shapes[0]} images, {types}, shape: {shapes[1:]}")
    
    unique_folderlist = _get_folders_from_imgs(imgs)
    l = l + unique_folderlist

    return "\n".join(l)


def figtitle_to_fig(imgs, fig: plt.Figure, axes: np.ndarray) -> None:
    """Makes a suptitle for figures"""
    figtitle = _figtitle_from_imgs(imgs)

    bbox_y = 1.05 if axes.shape[0] <= 2 else 1.01
    fig.suptitle(figtitle, ha="left", x=0.01, y=bbox_y, fontsize="large")


def imshow(
    imgs: l2Darrays | T.array,  # > Can be batch
    ### Subplots kws
    max_cols: int = 2,
    scalebar: bool = False,
    scalebar_kws: dict = dict(),
    ### Info persistant across batches of imgs
    imgs_raw: l2Darrays | T.array = None,
    i_in_batch: int = 0,  # > Index of first image in batch
    ### i/o
    save_as: str = None,
    ret=True,
    verbose=True,
    ### kws for _plot_img_to_ax()
    colorbar=True,
    share_cmap: bool = True,
    ### kws for ax.imshow()
    cmap: str = "gist_ncar",
    **ax_imshow_kws,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Show the images"""

    # === Prepare images ===
    ### Convert if single image
    if len(imgs.shape) == 2:
        imgs = l2Darrays([imgs])

    ### Make copies so scalebar isn't persistant
    _imgs_scalebar = imgs.copy()

    ### Keep raw images
    # - For calculating values for colorbar
    # - Extract information persistant across batches of imgs
    _imgs_raw = imgs.copy() if imgs_raw is None else imgs_raw

    ### Burn scalebar into _imgs
    if scalebar:
        _imgs_scalebar = scaleb.burn_scalebars(
            imgs=_imgs_scalebar, **scalebar_kws
        )

    # === Plot ===
    ### Calculate n_cols and n_rows
    n_cols = 1 if len(_imgs_scalebar) == 1 else max_cols
    n_rows = int(np.ceil(len(_imgs_scalebar) / n_cols))
    ### Init fig and axes
    fig, axes = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        figsize=(n_cols * 5, n_rows * 5),
        squeeze=False,
    )

    ### kws for ax.imshow()
    imshow_KWS = dict(cmap=cmap)
    imshow_KWS.update(ax_imshow_kws)

    ### Plot images into axes
    for i_in_ax, ax in enumerate(axes.flat):
        # > Prevent displaying empty axis when reaching end of imgs
        if i_in_ax >= len(_imgs_scalebar):
            ax.axis("off")
            continue
        # > Retrieve image with scalebar burned in
        img: np.ndarray = _imgs_scalebar[i_in_ax]
        # > Plot ax
        ax, cb = _plot_img_to_ax(
            ax=ax,
            img=img,
            imgs_raw=_imgs_raw,
            i_in_ax=i_in_ax,
            i_in_batch=i_in_batch,
            share_cmap=share_cmap,
            colorbar=colorbar,
            **imshow_KWS,
        )

    # === Edits ===
    ### Legend
    _plot_legend(fig, cb, n_cols, n_rows)

    ### Title
    figtitle_to_fig(_imgs_raw, fig=fig, axes=axes)

    ### Layout
    plt.tight_layout()

    # === I/O ===
    return plot_IO(
        fig=fig,
        axes=axes,
        ret=ret,
        save_as=save_as,
        verbose=verbose,
    )


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


def _test_imshow(imgs: np.ndarray):
    kws = dict(
        max_cols=2,
        scalebar=True,
        scalebar_kws=dict(
            pixel_length=0.05,  # !! must be provided when scalebar
        ),
        ret=False,
    )

    imshow(imgs[0], **kws)
    imshow(imgs[[0, 1, 2]], **kws)
    imshow(imgs[0:10:2], **kws)

    ### Check if scalebar is not burned it
    plt.imshow(imgs[0])
    plt.suptitle("no scalebar should be here")
    plt.show()


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


def save_figs_to_pdf(figs_axes, save_as):
    ### Add .pdf as suffix if no suffix is present
    save_as = _finalize_fname(save_as)
    with PdfPages(save_as) as pdf:
        for fig, _ in figs_axes:
            pdf.savefig(fig, bbox_inches="tight")


def plot_IO(
    fig: plt.Figure,
    axes: plt.Axes,
    ret: bool,
    save_as: str,
    verbose: bool = True,
) -> None | tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """Save figure to file"""
    if save_as:
        savefig(save_as=save_as, verbose=verbose)
    if ret:
        return fig, axes
    else:
        plt.show()


# %%
# == Batch Plotting ====================================================


def imshow_batched(
    imgs: T.array | l2Darrays,
    batch_size: int = 4,
    save_as: str = None,
    **imshow_kws,
):
    ### Info to persist across batches
    # - For calculating values for colorbar
    # - Extract information persistant across batches of imgs
    imgs_raw = imgs.copy()

    ### Plot in batches
    figs_axes = []
    for i_batch in range(0, len(imgs), batch_size):
        fig, axes = imshow(
            imgs=imgs[i_batch : i_batch + batch_size],
            imgs_raw=imgs_raw,
            i_in_batch=i_batch,
            ret=True,
            **imshow_kws,
        )
        figs_axes.append((fig, axes))

    if not save_as is None:
        save_figs_to_pdf(figs_axes, save_as)

    return figs_axes


def _test_imshow_batched(imgs):
    figs_axes = imshow_batched(
        imgs,
        batch_size=4,
        save_as="test",
    )
    # save_figs_to_pdf(figs_axes, "test.pdf")


if __name__ == "__main__":
    ### Test Plot l2Darrays
    _test_imshow(imgs=Z.imgs)
    # %%
    ### Test Batch-plot l2Darrays
    _test_imshow_batched(imgs=Z.imgs)
    # %%
    ### Test Plot np.ndarray
    _test_imshow(imgs=Z.imgs.asarray())
    #%% 
    ### Test Batch-plot np.ndarray
    _test_imshow_batched(imgs=Z.imgs.asarray())


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

    return plot_IO(
        fig=plt.gcf(),
        axes=plt.gca(),
        ret=ret,
        save_as=save_as,
        verbose=verbose,
    )

    # if not save_as is None:
    #     savefig(save_as=save_as, verbose=verbose)

    # if ret:
    #     return mip
    # else:
    #     plt.show()
