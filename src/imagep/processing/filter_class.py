""" A class for to implement all filters as methods """

# %%
from typing import Callable

import numpy as np

import time

import matplotlib.pyplot as plt

import scipy as sp
import skimage as ski
import skimage.morphology as morph

# > Local
import imagep._utils.utils as ut
import imagep.processing.filters_accelerated as filt_acc
import imagep.processing.filters as filt


# %%
# == Class Filter ===================================================
class Filter:
    def __init__(self, imgs: np.ndarray = None, verbose: bool = True):
        self.imgs = imgs
        self.verbose = verbose

    def __getitem__(self, val: slice):
        return self.imgs[val]

    def __iter__(self):
        return iter(self.imgs)

    #
    # == Denoise =======================================================
    def denoise(
        self,
        cached: bool = True,
        parallel: bool = True,
        n_cores: int = None,
        # inplace: bool = False, #!! not working
        **filter_kws,
    ):
        ### KWARGS
        # > Acceleration KWS
        n_cores = n_cores if n_cores else ut.cores_from_shape(self.imgs.shape)
        acc_KWS = dict(
            cached=cached,
            parallel=parallel,
            n_cores=n_cores,
        )
        # > Filter KWS
        filter_KWS = dict(imgs=self.imgs)  # > placeholder
        filter_KWS.update(filter_kws)

        ### Execute
        if self.verbose:
            return ut._messaged_execution(
                f=filt_acc.denoise,
                msg="Denoising",
                acc_KWS=acc_KWS,
                filter_KWS=filter_KWS,
            )
        else:
            return filt_acc.denoise(**filter_KWS, **acc_KWS)

    #
    # == entropy =====================================================
    def entropy(
        self,
        kernel_radius: int = 3,
        normalize: bool = True,
        cached: bool = True,
        parallel: bool = True,
        n_cores: int = None,
        **filter_kws,
    ) -> np.ndarray:
        """Performs entropy filter on image
        Kernel radius = 3 is best, since anything below will
        oversaturate regions, losing information. Could be used for
        segmentation, but edges are expanded, but this expansion doesn't
        change with radius
        """
        
        print(f"{type(self.imgs)=}")

        ### KWARGS
        # > Acceleration KWS
        n_cores = n_cores if n_cores else ut.cores_from_shape(self.imgs.shape)
        acc_KWS = dict(
            cached=cached,
            parallel=parallel,
            n_cores=n_cores,
        )
        # > Filter KWS
        filter_KWS = dict(
            imgs=self.imgs,
            kernel_radius=kernel_radius,
            normalize=normalize,
        )
        filter_KWS.update(filter_kws)

        ### Execute
        if self.verbose:
            return ut._messaged_execution(
                f=filt_acc.entropy,
                msg="Calculating local entropy",
                acc_KWS=acc_KWS,
                filter_KWS=filter_KWS,
            )
        else:
            return filt_acc.entropy(**filter_KWS, **acc_KWS)

    #
    # == Smoothen ======================================================
    def median(
        self,
        kernel_radius: int = 2,
        kernel_3D: bool = True,
        normalize: bool = True,
        cached: bool = True,
        **filter_kws,
    ) -> np.ndarray:
        """Performs median filter on image"""

        ### KWARGS
        # > Filter KWS
        filter_KWS = dict(
            imgs=self.imgs,
            kernel_radius=kernel_radius,
            kernel_3D=kernel_3D,
            normalize=normalize,
        )
        filter_KWS.update(filter_kws)

        ### Execute
        if self.verbose:
            return ut._messaged_execution(
                f=filt_acc.median,
                msg="Median filtering",
                acc_KWS=dict(cached=cached),
                filter_KWS=filter_KWS,
            )
        else:
            return filt_acc.median(**filter_KWS)

    #
    # == Blur ==========================================================

    def blur(
        self,
        sigma: float = 1,
        cross_z: bool = True,
        normalize: bool = True,
        **filter_kws,
    ) -> np.ndarray:
        """Blur image using a thresholding method"""

        ### KWARGS
        # > Filter KWS
        filter_KWS = dict(
            imgs=self.imgs,
            sigma=sigma,
            cross_z=cross_z,
            normalize=normalize,
        )
        filter_KWS.update(filter_kws)

        ### Execute
        if self.verbose:
            return ut._messaged_execution(
                f=filt.blur,
                msg="Gaussian Blurring",
                filter_KWS=filter_KWS,
            )
        else:
            return filt.blur(**filter_KWS)

    # !! == End Class ==================================================


# %%
# !! TESTDATA ==========================================================
from imagep.images.stack import Stack
from imagep._plots.imageplots import imshow

if __name__ == "__main__":
    # path_r = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    path_d = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"

    Z = Stack(path=path_d, verbose=True, x_µm=1.5 * 115.4)
    I = 6
    Z.imgs[I + 1] = 3  # > Change next image to test multidimensional filters
    T = Filter(imgs=Z.imgs, verbose=True)
    # %%
    ### Show NOT-denoised
    Z[[I, I + 1]].imshow()


# %%
# == Test Accelerated Transforms =======================================
def _get_test_kws_acc():
    """makes a list of all combinations of cached, parallel, n_cores"""
    ### Get all combinations of cached, parallel, n_cores
    cached = [True, False]
    parallel = [True, False]
    n_cores = [None, 2, 4]
    kws_list = [
        dict(cached=c, parallel=p, n_cores=n)
        for c in cached
        for p in parallel
        for n in n_cores
    ]
    return kws_list


def _test_denoise(T: Filter, I: int = 6):
    kws_list = _get_test_kws_acc()
    for kws in kws_list:
        print(kws)
        _imgs = T.denoise(**kws)
        imshow(_imgs[[I, I + 1]])
        plt.show()


def _test_entropy(T: Filter, I: int = 6):
    kws_list = _get_test_kws_acc()
    for kws in kws_list:
        print(kws)
        _imgs = T.entropy(**kws)
        imshow(_imgs[[I, I + 1]])
        plt.show()


if __name__ == "__main__":
    pass
    # _test_denoise(T, I) #:: uncomment
    # _test_entropy(T, I) #:: uncomment


# %%
# == Test Filters that aren't parallel =================================
def _test_median(T: Filter, I: int = 6):
    kws_list = [
        dict(kernel_radius=1, cross_z=True, normalize=True, cached=False),
        dict(kernel_radius=2, cross_z=True, normalize=True, cached=False),
        dict(kernel_radius=2, cross_z=False, normalize=True, cached=False),
        dict(kernel_radius=2, cross_z=True, normalize=False, cached=False),
        dict(kernel_radius=2, cross_z=True, normalize=False, cached=True),
    ]
    for kws in kws_list:
        print(kws)
        _imgs = T.median(**kws)
        imshow(_imgs[[I, I + 1]])
        plt.show()


def _test_blur(T: Filter, I: int = 6):
    kws_list = [
        dict(sigma=1, cross_z=True, normalize=True),
        dict(sigma=2, cross_z=True, normalize=True),
        dict(sigma=2, cross_z=False, normalize=True),
        dict(sigma=2, cross_z=True, normalize=False),
    ]
    for kws in kws_list:
        print(kws)
        _imgs = T.blur(**kws)
        imshow(_imgs[[I, I + 1]])
        plt.show()


if __name__ == "__main__":
    pass
    # _test_median(T, I) #:: uncomment
    _test_blur(T, I)


# %%
# !! Check from PreProcess ==============================================
if __name__ == "__main__":
    from imagep.processing.preprocess import PreProcess

    # %%
    hä

    Z2 = PreProcess(
        path=path_d,
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
    _imgs = Z2.filter.denoise(parallel=True)
    plt.imshow(_imgs[i])

    ### Denoising serial
    # > 14 Seconds for rough
    # > 17.0  seconds for detailed (1 workers)
    _imgs = Z2.filter.denoise(parallel=False)
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
