"""Utility functions for imagep"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import imagep.utils.scalebar as scalebar

#%%


def check_arguments(kws: dict, required_keys: list):
    """Check if all required keys are present in kws"""
    for k in required_keys:
        if not k in kws.keys():
            raise KeyError(f"Missing argument '{k}' in kws: {kws}")
        
        
#
# == Plots =========================================================

def mip(
    stack: np.ndarray,
    axis: int = 0,
    show=True,
    return_array=False,
    savefig: str = "mip.png",
    colormap: str = "gist_ncar",
) -> np.ndarray | None:
    """Maximum intensity projection across certain axis"""

    mip = stack.max(axis=axis)

    if show:
        plt.imshow(
            mip,
            cmap=colormap,
            interpolation="none",
        )
        # plt.show()

    if savefig:
        plt.imsave(fname=savefig, arr=mip, cmap=colormap, dpi=300)

    if return_array:
        return mip