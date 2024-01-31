""" A class for to implement all filters as methods """
# %%
import numpy as np

# > Since ThreadPoolExecutor is faster, we seem to be I/O bound
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import matplotlib.pyplot as plt

import scipy as sp
import skimage as ski
import skimage.morphology as morph

# > Local
import imagep._utils.utils as ut


# %%
### Number of cores to utilize
UTILIZE_CORES = ut.get_n_cores(utilize=0.33)  # > 33% is 3 of 8 cores!
UTILIZE_CORES
# %%
# == Parallelized Functions ============================================
### Multiprocessing can only pickle top-level functions


def _denoise_base(kws):
    return ski.restoration.denoise_nl_means(**kws)


def _denoise_parallel(imgs: np.ndarray) -> np.ndarray:
    ### List comprehensions are faster
    sigmas = [np.mean(ski.restoration.estimate_sigma(img)) for img in imgs]

    ### Collect arguments
    kws_list = []
    for img, sigma in zip(imgs, sigmas):
        kws_list.append(
            dict(
                image=img,
                h=0.8 * sigma,
                sigma=sigma,
                patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                fast_mode=True,
            )
        )

    ### Parallel Execution
    workers = ut.get_n_cores(utilize=UTILIZE_CORES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        imgs_iter = executor.map(_denoise_base, kws_list)

    _imgs = np.asarray(list(imgs_iter), dtype=imgs.dtype)
    return _imgs


if __name__ == "__main__":
    from imagep._imgs.imgs import Imgs

    # path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"

    Z = Imgs(path=path, verbose=True, x_µm=1.5 * 115.4)
    I = 6
    # %%
    ### Show NOT-denoised
    Z.imshow(slice=I)
    # %%
    ### Denoised
    # _imgs = _denoise_parallel(Z.imgs)
    # plt.imshow(_imgs[I])


# %%
# == Class Transform ===================================================
class Transform:
    def __init__(self, imgs: np.ndarray = None, verbose: bool = True):
        self.imgs = imgs
        self.verbose = verbose

    def __getitem__(self, val: slice):
        return self.imgs[val]

    def __iter__(self):
        return iter(self.imgs)

    #
    # == Denoise =======================================================

    def denoise(self, parallel=True) -> np.ndarray:
        ### Message, since it can take a while
        if self.verbose:
            if parallel:
                print(f"\tDenoising (workers={UTILIZE_CORES})...")
            else:
                print("\tDenoising...")

        ### Execute
        if parallel:
            _imgs = _denoise_parallel(imgs=self.imgs)
        else:
            _imgs = self._denoise()

        return _imgs

    def _denoise(self) -> np.ndarray:
        ### List comprehensions are faster
        sigmas = [
            np.mean(ski.restoration.estimate_sigma(img)) for img in self.imgs
        ]
        _imgs = [
            ski.restoration.denoise_nl_means(
                image=img,
                h=0.8 * sigma,
                sigma=sigma,
                patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                fast_mode=True,
            )
            for img, sigma in zip(self, sigmas)
        ]

        return np.array(_imgs, dtype=self.imgs.dtype)

    #
    # == Blur ==========================================================

    def blur(self, sigma: float = 1, normalize=True) -> np.ndarray:
        """Blur image using a thresholding method"""

        _imgs = ski.filters.gaussian(self.imgs, sigma=sigma)

        ### The max value is not 1 anymore
        if normalize:
            _imgs = _imgs / _imgs.max()

        return _imgs

    #
    # == Smoothen ======================================================

    def median(self, kernel_radius: int = 1, normalize=True) -> np.ndarray:
        """Performs median filter on image"""

        _imgs = np.zeros_like(self.imgs)

        kernel = ski.morphology.disk(
            radius=kernel_radius,
            # strict_radius=False,  # > extends radius by .5
        )

        for i, img in enumerate(self.imgs):
            img = ski.img_as_ubyte(img)

    #
    # == Variation =====================================================

    def entropy(self, imgs: np.ndarray=None, kernel_radius: int = 3, normalize=True) -> np.ndarray:
        """Performs entropy filter on image
        Kernel radius = 3 is best, since anything below will
        oversaturate regions, losing information. Could be used for
        segmentation, but edges are expanded, but this expansion doesn't
        change with radius
        """
        ### Use imgs if provided, otherwise use self.imgs 
        imgs = self.imgs if imgs is None else imgs

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


# !! ===================================================================

# %%
### Summarize info from images for testing
# def _print_info(imgs: np.ndarray):
#     just = ut.justify_str
#     print(f"{just('shape')} {imgs.shape}")
#     print(f"{just('dtype')} {imgs.dtype}")
#     print(f"{just('min, max')} {imgs.min(), imgs.max()}")


# %%
if __name__ == "__main__":
    from imagep.processing.preprocess import PreProcess

    Z2 = PreProcess(
        path=path,
        x_µm=1.5 * 115.4,
        verbose=True,
        denoise=True,
        # cache_preprocessing=False,
        subtract_bg=True,
        normalize=True,
    )

    # %%
    Z2.info

    # %%
    Z2.imshow(slice=I)


# %%
def _test_denoise_parallel(i=6):
    ### Denoising parallel
    # > 11 Seconds for rough (4 workers)
    # > 11 Seconds for detailed (4 workers)
    # > 10.8 Seconds for detailed (2 workers)
    ### ThreadPoolExecutor is faster than ProcessPoolExecutor!
    # > 3.5 Seconds! (2 workers) >> BEST!
    _imgs = Z2.transform.denoise(parallel=True)
    plt.imshow(_imgs[i])

    ### Denoising serial
    # > 14 Seconds for rough
    # > 17.0  seconds for detailed (1 workers)
    _imgs = Z2.transform.denoise(parallel=False)
    plt.imshow(_imgs[i])


if __name__ == "__main__":
    pass
    # _test_denoise_parallel(i=I)


# %%
# == Variation =========================================================
def _test_local_std(Z, i=6):
    _imgs = Z.transform.local_std()
    plt.imshow(_imgs[i])


def _test_entropy(Z, i=6):
    """ 
    Kernel radius = 2 is best, since r=1 is too grainy
    Edges are expanded, but don't change with radius
    """
    for r in range(1, 4, 1):
        print("kernel_radius: ", r)
        _imgs = Z.transform.entropy(kernel_radius=r)

        Z.imshow(slice=(i, i + 1), imgs=_imgs)
    

if __name__ == "__main__":
    # _test_local_std(Z2)
    _test_entropy(Z2, i=I)
