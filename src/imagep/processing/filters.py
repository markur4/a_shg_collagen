"""All filters that don't need acceleration"""
# %%
import numpy as np

import scipy as sp
import skimage as ski

# > Local
import imagep._utils.utils as ut


# %%
# !! Testdata ==========================================================
if __name__ == "__main__":
    from imagep.images.imgs import Imgs
    from imagep._plots.imageplots import imshow

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Imgs(path=path, verbose=True, x_µm=1.5 * 115.4)
    # > quick normalization
    Z.imgs = Z.imgs / Z.imgs.max()

    I = 6
    Z.imgs[I + 1] = .2  # > Chenge next image to test multidimensional filters
    imshow(Z.imgs[[I, I + 1]])


# %%
def blur(
    imgs: np.ndarray,
    sigma: float = 1,
    normalize=True,
    cross_z: bool = True,
    **filter_kws,
) -> np.ndarray:
    """Blur image using a thresholding method"""

    ### Collect kwargs
    kws = dict(
        sigma=sigma,
        # channel_axis=0, # > Color channel (expected as last axis)
        # truncate=sigma * 2,  # > Truncate filter at 2 sigma
    )
    kws.update(filter_kws)

    ### Apply 3D filter, blurring across z-axis leads to cross-talk !!!
    if cross_z:
        _imgs = ski.filters.gaussian(imgs, **kws)
    ### Apply 2D filter, no cross-talk in z-axis
    else:
        _imgs = [ski.filters.gaussian(img, **kws) for img in imgs]
        _imgs = np.array(_imgs, dtype=imgs.dtype)

    ### The max value is not 1 anymore
    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


if __name__ == "__main__":
    _imgs1 = blur(Z.imgs, sigma=1, cross_z=False, normalize=True)
    imshow(_imgs1[[I, I + 1]])



