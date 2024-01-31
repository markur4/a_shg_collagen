"""Plotting functions that display images"""

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path


def mip(
    imgs: np.ndarray,
    axis: int = 0,
    show=True,
    return_array=False,
    save: bool | str = False,
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

    if save:
        if not isinstance(save, str):
            raise ValueError("You must provide a filename for save")
        save = Path(save).with_suffix(".png")
        plt.imsave(fname=save, arr=mip, cmap=colormap)

    if return_array:
        return mip