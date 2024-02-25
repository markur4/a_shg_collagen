"""Utilities for plotting images"""
# %%
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

# > Local
import imagep.types as T
from imagep.arrays.l2Darrays import l2Darrays


#%%

def _get_folders_from_imgs(imgs: l2Darrays) -> list[str]:
    """Extracts folder names from imgs"""
    folders = []
    for img in imgs:
        if hasattr(img, "folder"):
            folders.append(img.folder)
    folders = list(set(folders))
    return folders


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
    l = l + _get_folders_from_imgs(imgs)
    ### Length, types
    l.append(f"Showing {shapes[0]} images, {types}, shape: {shapes[1:]}")
    # print(f"{l =}")

    return "\n".join(l)


def figtitle_to_fig(
    fig: plt.Figure,
    axes: np.ndarray,
    imgs: T.array | l2Darrays = None,
    title: str = "",
) -> None:
    """Makes a suptitle for figures"""

    title = title if title else _figtitle_from_imgs(imgs)


    bbox_y = 1.05 if axes.shape[0] <= 2 else 1.01
    fig.suptitle(title, ha="left", x=0.01, y=bbox_y, fontsize="large")


def _finalize_fname(fname: str) -> Path:
    """Add .pdf as suffix if no suffix is present"""
    if not isinstance(fname, str | Path):
        raise ValueError(
            f"You must provide a filename for save. Got: '{fname}'"
        )

    if "." in fname:
        return Path(fname)
    else:
        return Path(fname).with_suffix(".pdf")


def savefig(
    save_as: str | Path,
    verbose: bool = True,
) -> None:
    """Saves current plot to file"""
    save_as = _finalize_fname(save_as)

    ### Save
    plt.savefig(save_as, bbox_inches="tight")

    if verbose:
        print(f"Saved plot to: {save_as.resolve()}")


def save_figs_to_pdf(
    save_as: str | Path,
    figs_axes: tuple[plt.Figure, plt.Axes],
    # figs: list[plt.Figure],
    verbose: bool = True,
):
    ### Add .pdf as suffix if no suffix is present
    save_as = _finalize_fname(save_as)
    with PdfPages(save_as) as pdf:
        for fig, _ in figs_axes:
            pdf.savefig(fig, bbox_inches="tight")
    if verbose:
        print(f"Saved plots to: {save_as.resolve()}")


def return_plot(
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
        ### Subsequent code expects batched plotting
        return fig, axes
    else:
        plt.show()


def return_plot_batched(
    figs_axes: tuple[tuple[plt.Figure, np.ndarray[plt.Axes]]],
    # figs: list[plt.Figure],
    # axess: list[np.ndarray[plt.Axes]],
    ret: bool,
    save_as: str,
    verbose: bool = True,
) -> None | tuple[tuple[plt.Figure, np.ndarray[plt.Axes]]]:
    if save_as:
        save_figs_to_pdf(
            figs_axes=figs_axes,
            save_as=save_as,
            verbose=verbose,
        )
    if ret:
        return figs_axes
    else:
        plt.show()