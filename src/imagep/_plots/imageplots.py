"""Plotting functions that display images"""

# %%
from typing import TYPE_CHECKING
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from imagep._plots._plotutils import (
    figtitle_to_fig,
    return_plot,
    return_plot_batched,
)

# > local imports
import imagep._utils.utils as ut
import imagep.types as T
import imagep._plots.scalebar as scaleb
from imagep.arrays.l2Darrays import l2Darrays


# from imagep.images.imgs import Imgs


# %%
# == img to ax ==========================================================
def _colorbar_to_ax(
    ax_img: mpl.image,
    ax: plt.Axes,
    percentiles: tuple = None,
    min_max: tuple = None,
    fraction: float = 0.04,
    **colorbar_kws
) -> plt.colorbar:
    """Add colorbar to axes"""
    
    ### Gather kws
    kws = dict(
        mappable=ax_img,
        ax=ax,
        orientation="vertical",
        fraction=fraction,  # > Size colorbar relative to ax
    )
    kws.update(colorbar_kws)
    
    # > Colorbar
    cb = plt.colorbar(**kws)

    #### Metrics
    kws = dict(
        xmin=0,
        xmax=15,
        zorder=100,
        alpha=1,  # !! override
    )
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
    imgs_all: T.array,
    i_in_batch: int,
    ### kws for ax.imshow()
    **ax_imshow_kws,
) -> tuple[plt.Axes]:
    ### Get indices
    if hasattr(img, "index"):
        i_total = img.index[1]
        i_in_total = img.index[0]
        i_in_plot = i_in_batch + i_in_ax
        img_raw = imgs_all[i_in_plot]
    else:
        i_total = len(imgs_all) - 1
        i_in_total = i_in_batch + i_in_ax
        img_raw = imgs_all[i_in_total]

    ### Plot img to ax
    ax_img = ax.imshow(img, **ax_imshow_kws)

    ### Remove x- and y-axis
    ax.axis("off")

    ### Define color limits by image stack
    if share_cmap:
        min_max = (np.min(imgs_all), np.max(imgs_all))
        ax_img.set_clim(*min_max)

    ### Colorbar
    if colorbar:
        p = np.percentile(img_raw, [50, 75, 99])
        min_max_img = np.min(img_raw), np.max(img_raw)
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
def imshow(
    imgs: l2Darrays | T.array | list,  # > Can be batch
    ### Subplots kws
    max_cols: int = 2,
    scalebar: bool = True,
    scalebar_kws: dict = dict(),
    ### Info persistant across batches of imgs
    imgs_all: l2Darrays | T.array = None,
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
    ### Convert if list
    if isinstance(imgs, list):
        imgs = l2Darrays(imgs)

    ### Convert if single image
    if len(imgs.shape) == 2:
        imgs = l2Darrays([imgs])

    ### Keep all raw images
    # - For calculating values for colorbar
    # - Extract information persistant across batches of imgs
    _imgs_all = imgs if imgs_all is None else imgs_all

    ### Burn scalebar into _imgs
    if scalebar and hasattr(imgs[0], "pixel_length"):
        imgs = scaleb.burn_scalebars(imgs=imgs.copy(), **scalebar_kws)

    # === Plot ===
    ### Calculate n_cols and n_rows
    n_cols = 1 if len(imgs) == 1 else max_cols
    n_rows = int(np.ceil(len(imgs) / n_cols))
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
        if i_in_ax >= len(imgs):
            ax.axis("off")
            continue
        # > Retrieve image with scalebar burned in
        img: np.ndarray = imgs[i_in_ax]
        # > Plot ax
        ax, cb = _plot_img_to_ax(
            ax=ax,
            img=img,
            imgs_all=_imgs_all,
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
    figtitle_to_fig(imgs=imgs, fig=fig, axes=axes)

    ### Layout
    plt.tight_layout()

    # === return ===
    return return_plot(
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


# %%
# == Batch Plotting ====================================================
def imshow_batched(
    imgs: T.array | l2Darrays,
    batch_size: int = 4,
    ret: bool = False,
    save_as: str = None,
    verbose: bool = True,
    **imshow_kws,
):

    ### Plot in batches
    # figs, axess = [], []
    figs_axes = []
    for i_batch in range(0, len(imgs), batch_size):
        fig, axes = imshow(
            imgs=imgs[i_batch : i_batch + batch_size],
            imgs_all=imgs,
            i_in_batch=i_batch,
            ret=True,  # > must return fig, axes
            **imshow_kws,
        )
        # figs.append(fig)
        # axess.append(axes)
        figs_axes.append((fig, axes))

    return return_plot_batched(
        figs_axes=figs_axes,
        # figs=figs,
        # axess=axess,
        ret=ret,
        save_as=save_as,
        verbose=verbose,
    )


# %%
# == mip ===============================================================
def _multi_mip(imgs: np.ndarray, colormap: str = "gist_ncar") -> None:
    """Plots a 3D volume as maximum intensity projections along each
    axis in a pretty way, assigning each axis to a subplots.
    """

    ### Make projections:
    Z = imgs.max(axis=0)
    Y = imgs.max(axis=1)
    X = imgs.max(axis=2).T  # > rotate by 90 degrees

    ### Init Plot
    z_length = X.shape[1] / X.shape[0]
    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        width_ratios=[1, z_length],
        height_ratios=[1, z_length],
        gridspec_kw=dict(
            hspace=0.1,
            wspace=0.001,
        ),
    )

    ### kws for ax.imshow()
    kws = dict(
        cmap=colormap,
        interpolation="none",
    )
    _ax: plt.Axes  # > Declare type

    ### Z Projection (default)
    _ax = axes[0, 0]
    _ax.imshow(Z, **kws)
    _ax.set_title("z-Projection", fontsize="medium")
    _ax.set_ylabel("y")
    # > Remove x-axis labels
    _ax.set_xticklabels([])

    ### Y Projection (bottom)
    _ax = axes[1, 0]
    _ax.imshow(Y, **kws)
    # _ax.set_title("y-Projection", loc="right")
    _ax.set_xlabel("x\ny-Projection")
    _ax.set_ylabel("z")

    ### X Projection (right)
    _ax = axes[0, 1]
    # > rotate axis by 90 degrees so that it's displayed as y-projection
    _ax_img = _ax.imshow(X, **kws)
    _ax.set_title("x-Projection", fontsize="medium")
    # > Remove y-axis
    _ax.set_yticklabels([])
    
    # > Add colorbar
    cb = _colorbar_to_ax(_ax_img, _ax, fraction=.2, )
    _ax.set_xlabel("z")

    ### Empty
    _ax = axes[1, 1]
    _ax.axis("off")

    # === Edits ===

    ### Shift x-projection to the left
    pos_z = axes[0, 0].get_position()
    pos_x = axes[0, 1].get_position()
    new_pos_x = [
        pos_z.x0 + pos_z.width + 0.02,
        pos_z.y0,
        pos_x.width,
        pos_z.height,
    ]
    axes[0, 1].set_position(new_pos_x)

    ### Resize colorbar
    pos_cb = cb.ax.get_position()
    new_pos_cb = [
        0.9,
        0.1,
        pos_cb.width,
        0.8,
    ]
    cb.ax.set_position(new_pos_cb)


def mip(
    imgs: np.ndarray,
    axis: int | str = 0,
    colormap: str = "gist_ncar",
    ret=False,
    save_as: str = None,
    verbose=True,
) -> np.ndarray | None:
    """Maximum intensity projection across certain axis
    :param imgs: 3D array
    :param axis: Axis to project. If 'all', subplots are created for
        each axis
    :type axis: int|str
    :param colormap: Colormap
    """

    if isinstance(axis, int):
        mip = imgs.max(axis=axis)
        plt.imshow(
            mip,
            cmap=colormap,
            interpolation="none",
        )
    elif axis == "all" and len(imgs.shape) == 3:
        _multi_mip(imgs=imgs, colormap=colormap)

    else:
        raise ValueError(f"Invalid axis: '{axis}', must be int or 'all'")

    return return_plot(
        fig=plt.gcf(),
        axes=plt.gca(),
        ret=ret,
        save_as=save_as,
        verbose=verbose,
    )


# %%
# :: Test ==============================================================
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


def _test_imshow_batched(imgs):
    figs_axes = imshow_batched(
        imgs,
        batch_size=4,
        save_as="test",
        ret=True,
    )


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
    print(f"{Z.imgs[2].index =}")

    # %%
    ### Test Plot l2Darrays
    _test_imshow(imgs=Z.imgs)
    # %%
    ### Test Batch-plot l2Darrays
    _test_imshow_batched(imgs=Z.imgs)
    # %%
    ### Test Plot np.ndarray
    _test_imshow(imgs=Z.imgs.asarray())
    # %%
    ### Test Batch-plot np.ndarray
    _test_imshow_batched(imgs=Z.imgs.asarray())
