"""Here are all filters:
- all static and on global namespace
- all have similar interface
- If accelerated, then parallelized and cached
"""
# %%
import os
from pprint import pprint
import numpy as np

import time

from concurrent.futures import ThreadPoolExecutor

import scipy as sp
import skimage as ski

# > local imports
import imagep._utils.utils as ut
from imagep._utils.subcache import SubCache


# %%
# == CACHE =============================================================
# > Location
location = os.path.join(os.path.expanduser("~"), ".cache")

### Subcache
CACHE_FILTERS = SubCache(
    location=location,
    subcache_dir="filters",
    verbose=True,
    compress=9,
    bytes_limit="3G",  # > 3GB of cache, keeps only the most recent files
)
# %%
pprint(CACHE_FILTERS.kwargs)


# %%
# == CORES =============================================================
UTILIZE_CORES = ut.get_n_cores(utilize=0.33)  # > 33% is 3 of 8 cores!


# %%
#!! TESTDATA ===========================================================
if __name__ == "__main__":
    from imagep._imgs.imgs import Imgs
    from imagep._plottools.imageplots import imshow

    # path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"

    Z = Imgs(path=path, verbose=True, x_µm=1.5 * 115.4)
    Z.imgs = Z.imgs / Z.imgs.max()  # > quick normalization
    I = 6
    # %%
    ### Show NOT-denoised
    Z[I].imshow()

# %%
# == DENOISE ===========================================================

DENOISE_KWS = dict(
    patch_size=5,  # 5x5 patches
    patch_distance=6,  # 13x13 search area
    fast_mode=True,
)


def _denoise_sequential(imgs: np.ndarray) -> np.ndarray:
    ### List comprehensions are faster
    sigmas = [np.mean(ski.restoration.estimate_sigma(img)) for img in imgs]
    _imgs = [
        ski.restoration.denoise_nl_means(
            image=img,
            h=0.8 * sig,
            sigma=sig,
            **DENOISE_KWS,
        )
        for img, sig in zip(imgs, sigmas)
    ]
    return np.array(_imgs, dtype=imgs.dtype)


def _denoise_parallel_base(kwargs):
    return ski.restoration.denoise_nl_means(**kwargs)


def _denoise_parallel(imgs: np.ndarray) -> np.ndarray:
    ### List comprehensions are faster
    sigmas = [np.mean(ski.restoration.estimate_sigma(img)) for img in imgs]

    ### Collect arguments
    kws_list = []
    for img, sig in zip(imgs, sigmas):
        kws_list.append(
            dict(
                image=img,
                h=0.8 * sig,
                sigma=sig,
                **DENOISE_KWS,
            )
        )

    ### Parallel Execution
    workers = ut.get_n_cores(utilize=UTILIZE_CORES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        _imgs_iter = executor.map(_denoise_parallel_base, kws_list)

    _imgs = np.asarray(list(_imgs_iter), dtype=imgs.dtype)
    return _imgs


def _denoise(imgs: np.ndarray, parallel: bool = True) -> np.ndarray:
    """Choose parallel or sequential execution"""
    if parallel:
        f = _denoise_parallel
    else:
        f = _denoise_sequential

    _imgs = f(imgs=imgs)

    return _imgs


def denoise(imgs: np.ndarray, parallel: bool = True, cached=True) -> np.ndarray:
    """Implements caching"""
    t1 = time.time()
    m = "\tDenoising"

    ### Execute
    if cached:
        print(m + " (checking cache) ...")
        f = CACHE_FILTERS.subcache(_denoise)
        _imgs = f(imgs=imgs, parallel=parallel)
    else:
        print(m + " ...")
        _imgs = _denoise(imgs=imgs, parallel=parallel)
    dt = time.time() - t1

    print(f"\tDenoising done ({dt:.2f} s)")

    return _imgs


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
    _imgs3 = denoise(Z.imgs, parallel=True, cached=True)
    imshow(_imgs3[I])


# %%
# == Local Entropy =====================================================
def _entropy_sequential(
    imgs: np.ndarray,
    kernel_radius: int = 3,
    normalize=True,
) -> np.ndarray:
    """Performs entropy filter on image
    Kernel radius = 3 is best, since anything below will
    oversaturate regions, losing information. Could be used for
    segmentation, but edges are expanded, but this expansion doesn't
    change with radius
    """

    kernel = ski.morphology.disk(
        radius=kernel_radius,
        # strict_radius=False,  # > extends radius by .5
    )

    ### Filter
    _imgs = np.zeros_like(imgs)
    for i, img in enumerate(imgs):
        img = ski.img_as_ubyte(img)  # > Rankfilters require uint8
        _imgs[i] = ski.filters.rank.entropy(
            img,
            footprint=kernel,
        )

    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


def _entropy_parallel_base(kwargs):
    return ski.filters.rank.entropy(**kwargs)


def _entropy_parallel(
    imgs, kernel_radius: int = 3, normalize=True
) -> np.ndarray:
    """Performs entropy filter on image parallel"""

    kernel = ski.morphology.disk(radius=kernel_radius)

    ### Collect arguments
    kws_list = []
    for img in imgs:
        kws_list.append(
            dict(
                image=ski.img_as_ubyte(img),
                footprint=kernel,
            )
        )

    ### Parallel Execution
    workers = ut.get_n_cores(utilize=UTILIZE_CORES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        _imgs_iter = executor.map(_entropy_parallel_base, kws_list)

    _imgs = np.asarray(list(_imgs_iter), dtype=imgs.dtype)

    if normalize:
        _imgs = _imgs / _imgs.max()

    return _imgs


def _entropy(
    imgs: np.ndarray,
    kernel_radius: int = 3,
    normalize=True,
    parallel: bool = True,
) -> np.ndarray:
    """Choose parallel or sequential execution"""
    if parallel:
        f = _entropy_parallel
    else:
        f = _entropy_sequential

    _imgs = f(
        imgs=imgs,
        kernel_radius=kernel_radius,
        normalize=normalize,
    )

    return _imgs


def entropy(
    imgs: np.ndarray,
    kernel_radius: int = 3,
    normalize=True,
    parallel: bool = True,
    cached=True,
) -> np.ndarray:
    """Implements caching"""
    t1 = time.time()
    m = "\tLocal Entropy"

    ### Execute
    if cached:
        print(m + " (checking cache) ...")
        f = CACHE_FILTERS.subcache(_entropy)
        _imgs = f(
            imgs=imgs,
            kernel_radius=kernel_radius,
            normalize=normalize,
            parallel=parallel,
        )
    else:
        print(m + " ...")
        _imgs = _entropy(
            imgs=imgs,
            kernel_radius=kernel_radius,
            normalize=normalize,
            parallel=parallel,
        )
    dt = time.time() - t1

    print(f"\tEntropy done ({dt:.2f} s)")

    return _imgs


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
    _img3_bg = _imgs3 - (_imgs3.min() * 6)
    #>  replace negative with zero
    _img3_bg[_img3_bg < 0] = 0
    print(_img3_bg.min(), _img3_bg.max())
    imshow(_img3_bg[I])
    #%%
    ### Entropy parallel + cached (of denoised + bg subtracted image)
    _imgs7 = entropy(_img3_bg, parallel=True, cached=True)
    imshow(_imgs7[1:10:2])
