"""Here are all filters:
- all static and on global namespace
- all have similar interface
- If accelerated, then parallelized and cached
"""

# %%
import os
from pprint import pprint
import numpy as np

from concurrent.futures import ThreadPoolExecutor

import scipy as sp
import skimage as ski

# > local imports
# import imagep._configs.caches as caches
from imagep._configs.caches import FILTERS
import imagep._configs.rc as rc
import imagep._utils.utils as ut
import imagep._utils.types as T
from imagep.images.l2Darrays import l2Darrays
from imagep.images.mdarray import mdarray
import imagep._utils.metadata as meta


# %%
# == CHECK CACHE =======================================================
if __name__ == "__main__":
    pprint(FILTERS.get_cached_inputs())


# %%
#!! TESTDATA ===========================================================
if __name__ == "__main__":
    from imagep.images.stack import Stack
    from imagep._plots.imageplots import imshow

    # path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"

    Z = Stack(path=path, verbose=True, x_µm=1.5 * 115.4)
    Z.imgs = Z.imgs / Z.imgs.max()  # > quick normalization
    I = 6
    # %%
    ### Show NOT-denoised
    Z[I].imshow()

    # %%
    ### Are float16 enough to fully encode the image?
    print(Z.imgs.dtype)


# %%
# !! ===================================================================
# !! DENOISE ===========================================================
def _collect_denoise_filter_kws(imgs: np.ndarray) -> list[dict]:
    """Collects denoise kws for each image"""
    sigmas = [np.mean(ski.restoration.estimate_sigma(img)) for img in imgs]
    return [
        dict(
            image=img,
            h=0.8 * sig,
            sigma=sig,
            patch_size=5,  # 5x5 patches
            patch_distance=6,  # 13x13 search area
            fast_mode=True,
        )
        for img, sig in zip(imgs, sigmas)
    ]


# == Denoise Sequential ==


#!! Always decorate the main filter, or the numpy output is cached!
# @meta.preserve_metadata()
def _denoise_sequential(imgs: np.ndarray) -> np.ndarray:
    kws_list = _collect_denoise_filter_kws(imgs)
    _imgs = [ski.restoration.denoise_nl_means(**kws) for kws in kws_list]
    # return np.array(_imgs, dtype=imgs.dtype)
    return l2Darrays(_imgs, dtype=imgs.dtype)


# == Denoise Parallel ==
def _denoise_parallel_base(kwargs):
    return ski.restoration.denoise_nl_means(**kwargs)


def _denoise_parallel(imgs: T.array, n_cores: int = 2) -> np.ndarray:

    ### List comprehensions are faster
    sigmas = [np.mean(ski.restoration.estimate_sigma(img)) for img in imgs]

    ### Collect arguments
    kws_list = _collect_denoise_filter_kws(imgs)

    ### Parallel Execution
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        _imgs_iter = executor.map(_denoise_parallel_base, kws_list)

    # _imgs = np.asarray(list(_imgs_iter), dtype=imgs.dtype)
    _imgs = l2Darrays(_imgs_iter, dtype=imgs.dtype)
    return _imgs


# == Denoise MAIN ==
def _denoise(
    imgs: np.ndarray, parallel: bool = True, n_cores: int = 2
) -> np.ndarray:
    """Choose parallel or sequential execution"""
    if parallel:
        return _denoise_parallel(imgs=imgs, n_cores=n_cores)
    else:
        return _denoise_sequential(imgs=imgs)


@meta.preserve_metadata()
def denoise(
    imgs: np.ndarray,
    cached: bool = True,
    parallel: bool = True,
    n_cores: int = 2,
) -> np.ndarray:
    """Implements caching"""
    kws = dict(parallel=parallel, n_cores=n_cores)
    if cached:
        f = FILTERS.subcache(_denoise, ignore=["parallel", "n_cores"])
        return f(imgs=imgs, **kws)
    else:
        return _denoise(imgs=imgs, **kws)


if __name__ == "__main__":
    pass
    ### sequential
    # > 16 seconds
    # _imgs1 = _denoise_sequential(Z.imgs)
    # %%
    # imshow(_imgs1[I])

    # %%
    ### parallel
    # > 3.4 seconds
    # _imgs2 = _denoise_parallel(Z.imgs)

    # %%
    ### parallel + cached
    # > 1st run: 6.1 seconds
    # > 2nd run: 0.3 seconds
    # > 3rd run: 0.3 seconds
    _imgs3 = denoise(Z.imgs, parallel=True, cached=True, n_cores=4)
    imshow(_imgs3[I])


# %%
# !! ===================================================================
# !! Local Entropy =====================================================
def _collect_entropy_filter_kws(
    imgs: np.ndarray, kernel_radius=3
) -> list[dict]:
    """Collects entropy kws for each image"""
    kernel = ski.morphology.disk(
        radius=kernel_radius,
        strict_radius=False,  # > extends radius by .5
    )
    m = imgs.max()
    return [
        dict(
            image=ski.img_as_ubyte(img / m),  # > requires images in [0,1]
            footprint=kernel,
        )
        for img in imgs
    ]


# == Entropy Sequential ==
def _entropy_sequential(
    imgs: np.ndarray,
    kernel_radius: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """Performs entropy filter on image
    Kernel radius = 3 is best, since anything below will
    oversaturate regions, losing information. Could be used for
    segmentation, but edges are expanded, but this expansion doesn't
    change with radius
    """

    kws_list = _collect_entropy_filter_kws(imgs, kernel_radius=kernel_radius)
    _imgs = [ski.filters.rank.entropy(**kws) for kws in kws_list]
    _imgs = l2Darrays(_imgs, dtype=imgs.dtype)

    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


# == Entropy Parallel ==
def _entropy_parallel_base(kwargs):
    return ski.filters.rank.entropy(**kwargs)


def _entropy_parallel(
    imgs,
    kernel_radius: int = 3,
    normalize=True,
    n_cores: int = 2,
) -> np.ndarray:
    """Performs entropy filter on image parallel"""

    ### Collect arguments
    kws_list = _collect_entropy_filter_kws(imgs, kernel_radius=kernel_radius)

    ### Parallel Execution
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        _imgs_iter = executor.map(_entropy_parallel_base, kws_list)

    # _imgs = np.asarray(list(_imgs_iter), dtype=imgs.dtype)
    _imgs = l2Darrays(list(_imgs_iter), dtype=imgs.dtype)

    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


# == Entropy MAIN ==
def _entropy(
    imgs: np.ndarray,
    kernel_radius: int = 3,
    normalize: bool = True,
    parallel: bool = True,
    n_cores: int = 2,
) -> np.ndarray:
    """Choose parallel or sequential execution"""
    kws = dict(
        imgs=imgs,
        kernel_radius=kernel_radius,
        normalize=normalize,
    )
    if parallel:
        return _entropy_parallel(n_cores=n_cores, **kws)
    else:
        return _entropy_sequential(**kws)


@meta.preserve_metadata()
def entropy(
    imgs: np.ndarray,
    kernel_radius: int = 3,
    normalize: bool = True,
    cached: bool = True,
    parallel: bool = True,
    n_cores: int = 2,
) -> np.ndarray:
    """Implements caching"""
    kws = dict(
        kernel_radius=kernel_radius,
        normalize=normalize,
        parallel=parallel,
        n_cores=n_cores,
    )
    if cached:
        f = FILTERS.subcache(_entropy, ignore=["parallel", "n_cores"])
        return f(imgs=imgs, **kws)
    else:
        return _entropy(imgs=imgs, **kws)


if __name__ == "__main__":
    pass
    # %%
    ### sequential
    # > 7 seconds
    _imgs4 = _entropy_sequential(Z.imgs)
    imshow(_imgs4[I])

    # %%
    ### parallel
    # > 1.1 seconds
    _imgs5 = _entropy_parallel(Z.imgs)

    # %%
    ### Entropy parallel + cached (of raw image)
    # > 1st run: 5.23 seconds
    # > 2nd run: 0.3 seconds
    # > 3rd run: 0.3 seconds
    _imgs6 = entropy(Z.imgs, parallel=True, cached=True)
    imshow(_imgs6[I])
    # %%
    ### quick bg subtraction
    _imgs3_bg = _imgs3 - (_imgs3.min() * 6)
    # >  replace negative with zero
    _imgs3_bg[_imgs3_bg < 0] = 0
    print(_imgs3_bg.min(), _imgs3_bg.max())
    imshow(_imgs3_bg[I])
    # %%
    ### Entropy parallel + cached (of denoised + bg subtracted image)
    _imgs7 = entropy(_imgs3_bg, parallel=True, cached=True)
    imshow(_imgs7[I])

# %%
# !! == End Class ==================================================
# !! Median ============================================================


#!! Always decorate the main filter, or the numpy output is cached!
# @meta.preserve_metadata()
def _median(
    imgs: np.ndarray,
    kernel_radius: int = 2,
    kernel_3D: bool = False,
    normalize=False,
    **filter_kws,
) -> np.ndarray:
    """Performs median filter on image"""

    ### Apply 3D or 2D filter
    if kernel_3D:
        axes = None  # > Cross-talk in z-axis
        kernel = ski.morphology.ball(radius=kernel_radius)
    else:
        axes = (1, 2)  # > No cross-talk in z-axis
        kernel = ski.morphology.disk(radius=kernel_radius)

    ### Collect kws
    kws = dict(
        footprint=kernel,
        # axes=axes,
        **filter_kws,
    )

    ### Execute
    if kernel_3D:
        _imgs = ski.filters.rank.median(imgs, axes=axes, **kws)
    else:
        _imgs = [ski.filters.rank.median(img, **kws) for img in imgs]

    # _imgs = np.array(_imgs, dtype=imgs.dtype)
    _imgs = l2Darrays(_imgs, dtype=imgs.dtype)

    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


@meta.preserve_metadata()
def median(
    imgs: np.ndarray,
    kernel_radius: int = 2,
    kernel_3D: bool = True,
    normalize=False,
    cached: bool = True,
    **filter_kws,
) -> np.ndarray:
    """Implements caching"""
    kws = dict(
        kernel_radius=kernel_radius,
        kernel_3D=kernel_3D,
        normalize=normalize,
        **filter_kws,
    )
    if cached:
        f = FILTERS.subcache(_median, ignore=[])
        return f(imgs=imgs, **kws)
    else:
        return _median(imgs=imgs, **kws)
