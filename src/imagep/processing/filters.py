"""All filters that don't need acceleration"""

# %%
import numpy as np

import scipy as sp
import skimage as ski

# > Local
import imagep._utils.utils as ut
from imagep.images.mdarray import mdarray
from imagep.images.l2Darrays import l2Darrays
import imagep._utils.metadata as meta


# %%
# !! Testdata ==========================================================
if __name__ == "__main__":
    from imagep.images.stack import Stack
    from imagep._plots.imageplots import imshow

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Stack(
        data=path,
        verbose=True,
        fname_extension="txt",
        imgname_position=1,
        pixel_length=(1.5 * 115.4) / 1024,
    )
    # > Check type
    print(type(Z.imgs))
    print(type(Z.imgs[0]))
    # > quick normalization
    Z.imgs = Z.imgs / Z.imgs.max()
    # Z.imgs = Z.imgs / Z.imgs

    # > Check type
    print(type(Z.imgs))
    print(type(Z.imgs[0]))

    # %%
    Z.imgs

    # %%

    I = 6
    Z.imgs[I + 1] = 0.2  # > Chenge next image to test multidimensional filters
    imshow(Z.imgs[[I, I + 1]])


# %%
@meta.preserve_metadata()
def blur(
    imgs: np.ndarray,
    sigma: float = 1,
    normalize=True,
    kernel_3D: bool = True,
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
    if kernel_3D:
        _imgs = ski.filters.gaussian(imgs, **kws)
    ### Apply 2D filter, no cross-talk in z-axis
    else:
        # print(type(imgs[0]))
        _imgs = [ski.filters.gaussian(img, **kws) for img in imgs]
        # print(type(_imgs[0]))
        # _imgs = np.array(_imgs, dtype=imgs.dtype)
        _imgs = l2Darrays(_imgs, dtype=imgs.dtype)

    ### The max value is not 1 anymore
    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


if __name__ == "__main__":
    _imgs1 = blur(Z.imgs, sigma=1, kernel_3D=False, normalize=True)
    imshow(_imgs1[[I, I + 1]])

    # %%
    _imgs2 = blur(Z.imgs, sigma=1, kernel_3D=True, normalize=True)
    imshow(_imgs2[[I, I + 1]])

    # %%
    ### metadata preserved?
    print("NAME:", _imgs2[0].name)
    _imgs2
