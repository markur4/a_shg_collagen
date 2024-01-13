#
# %%

from pprint import pprint

from collections import OrderedDict

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

### increase display dpi of matplotlib
plt.rcParams["figure.dpi"] = 300
import seaborn as sns

### Analysis
import scipy as sp

# from scipy import ndimage

import skimage.morphology as morph
from skimage import draw

# from skimage import data
# from skimage.util import invert

from skimage import filters

# > Internal
from mkshg.import_and_preprocess import PreProcess

# %%


class Diameter(PreProcess):
    def __init__(self, **preprocess_kwargs):
        ### Don't burn a scalebar
        preprocess_kwargs["scalebar_micrometer"] = False

        super().__init__(**preprocess_kwargs)

        # TODO: Implement Segmentation
        self.st_segmented = self.segmentation()

        ### Initialize stacks for processing
        self.st_distance = np.zeros(self.st_segmented.shape)
        self.st_skeleton = np.zeros(self.st_segmented.shape)
        self.st_skeldist = np.zeros(self.st_segmented.shape)
        self.st_intersects = np.zeros(self.st_segmented.shape)
        self.st_intersects_dil = np.zeros(self.st_segmented.shape)

        ### Execute transform pipeline
        # self.measure_fiber_width()

    def measure_fiber_width(
        self,
        # sigma: float = 1,
        skelet_method=None,
        **edt_kws,
    ) -> np.ndarray:
        """Pipeline for measuring fiber width"""
        ### Start with segmented stack
        stack = self.st_segmented

        ### Distance Transform
        self.st_distance = self.distance_transform(
            stack,
            **edt_kws,
        )

        ### Skeletonize
        self.st_skeleton = self.skeletonize(
            # self.st_distance,
            stack,
            method=skelet_method,
        )

        ### Intersections
        # > Detect Intersections
        self.st_intersects = self.detect_intersections(self.st_skeleton)
        self.st_intersects_dil = self.dilate_intersections(
            intersects=self.st_intersects,
            distance_transform=self.st_distance,
        )

        ### Apply distances to Skeleton
        self.st_skeldist = self.st_skeleton * self.st_distance
        # > Removing Intersections leaves skeleton with gaps, intensity
        # ' of skeleton is half the diameter
        self.st_skeldist_removed = self.remove_intersections(
            skeleton=self.st_skeldist,
            intersects_dil=self.st_intersects_dil,
        )

        ### Calculate Diameters
        self.diameters = self.calc_diameters(skeldist=self.st_skeldist_removed)
        self.diameters_flat = np.concatenate(self.diameters)

    #
    # == Tools =========================================================

    def segmentation(self) -> np.ndarray:
        """Arbitrary segmentation, makes binary mask: replaces
        background with 0 and foreground with 1"""
        stack = self.stack > 0

        ### Improve quality
        for i, img in enumerate(stack):
            # > Closing + opening to remove small holes and salt/pepper
            # > Closing = dilation, then erosion
            # > Opening = erosion, then dilation
            # stack[i] = morph.closing(img)
            # stack[i] = morph.opening(stack[i])
            # > Smoothen edges to reduce false intersections
            stack[i] = sp.signal.medfilt(
                stack[i].astype(np.uint8),
                kernel_size=5,
            )
            stack[i] = stack[i].astype(bool)

        return stack

    @staticmethod
    def distance_transform(S: np.ndarray, **edt_kws) -> np.ndarray:
        """performs euclidean distance transform on each image in
        stack"""

        ### Calculate distance transform
        stack = np.zeros(S.shape)  # Init empty stack
        for i, img in enumerate(S):
            stack[i] = sp.ndimage.distance_transform_edt(img, **edt_kws)

        return stack

    @staticmethod
    def skeletonize(S: np.ndarray, method=None) -> np.ndarray:
        """Skeletonize each image in stack"""

        ### Calculate skeleton
        stack = np.zeros(S.shape)
        for i, img in enumerate(S):
            stack[i] = morph.skeletonize(img, method=method)

        return stack

    @staticmethod
    def detect_intersections(skeleton: np.ndarray) -> np.ndarray:
        """Remove intersections from skeleton"""
        kernel = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )

        ### Convolve the skeleton image with the kernel
        stack = np.zeros(skeleton.shape)  # Init empty stack
        for i, img in enumerate(skeleton):
            convolved = sp.ndimage.convolve(
                img,
                kernel,
                mode="constant",
                cval=0.0,
            )

            ### The intersections are the points with a value greater than 3
            intersections = (convolved > 3) & (img == 1)

            stack[i] = intersections

        return stack

    @staticmethod
    def dilate_intersections(
        intersects: np.ndarray, distance_transform: np.ndarray
    ) -> np.ndarray:
        """Dilates intersections to a disk shaped size, where the size
        is determined by the value of the distance transform at that
        intersection"""
        ### Get image of distance values where the intersects are
        st_inters_dist = np.where(
            intersects == 1,
            distance_transform,
            0,
        )

        stack = np.zeros_like(st_inters_dist)
        for i, img in enumerate(st_inters_dist):
            ### Get the coordinates of the intersects
            coords = np.argwhere(img > 0)

            ### For each intersect, add a disk to the result
            for x, y in coords:
                # > Set disk radius to the value of the intersection
                radius = int(img[x, y])
                rr, cc = draw.disk((x, y), radius, shape=img.shape)
                stack[i, rr, cc] = img[x, y]

        return stack

    @staticmethod
    def remove_intersections(
        skeleton: np.ndarray, intersects_dil: np.ndarray
    ) -> np.ndarray:
        """Removes the dilated intersections from the skeleton.
        Removing Intersections leaves skeleton with gaps, intensity of
        skeleton is half the diameter"""

        stack = np.zeros(skeleton.shape)  # Init empty stack
        for i, skel in enumerate(skeleton):
            skel_rem = np.where(intersects_dil[i] > 0, np.nan, skel)
            # > Replace zeros with nan
            skel_rem = np.where(skel_rem == 0, np.nan, skel_rem)

            stack[i] = skel_rem

        return stack

    @staticmethod
    def calc_diameters(skeldist: np.ndarray) -> np.ndarray:
        """Calculate the diameter of each fiber in each image"""

        diameters = []
        for img in skeldist:
            ### Get all the values of the skeleton
            values = img[~np.isnan(img)] * 2

            ### Calculate the diameter of each fiber
            diameters.append(values)

        return diameters

    # == Plotting ======================================================

    def plot_histogram(
        self,
        I: int = None,
        bins: int = 50,
        in_µm=True,
        **kws,
    ):
        """Plot histogram of fiber width"""

        diameters = self.diameters_flat if I is None else self.diameters[I]

        ### Convert to µm
        if in_μm:
            diameters = diameters * self.pixel_size

        # bins_auto = int(round(max(diameters)))
        # bins = bins_auto if bins == "max" else bins

        KWS = dict(
            bins=bins,
            log=True,
        )
        KWS.update(kws)

        plt.hist(diameters, **KWS)

        ### Edit plot
        plt.xlabel("Fiber width [µm]")
        plt.ylabel("Count of Pixels in Skeleton")

    # == Visualization of Intermediates ================================

    def plot_overlay(self, img="stack", I: int = 0, skelet_alpha=1):
        """Plot overlay of skeleton and segmented image"""

        ### Image
        if img == "stack":
            img = self.stack[I]
        elif img == "segmented":
            img = self.st_segmented[I]
        elif img == "distance":
            img = self.st_distance[I]

        plt.imshow(img, interpolation="none", cmap="gray")

        ### Skeleton
        # > Mask
        skel = self.st_skeldist[I]
        skel = np.ma.masked_where(skel == 0, skel)

        plt.imshow(
            skel, cmap="spring", interpolation="none", alpha=skelet_alpha
        )

    def plot_masked_by_segmentation(self, I: int = 0, alpha: float = 0.4):
        """Plot an image masked by segmentation"""
        img = self.stack[I]
        seg = self.st_segmented[I]

        # > Mask
        masked = np.ma.masked_where(seg == 0, img)

        plt.imshow(masked, interpolation="none", cmap="gray", alpha=alpha)

    def plot_skeleton_and_intersections(
        self,
        I: int = 0,
        alpha: float = 0.4,
        skel="skeldist_removed",
    ):
        """Plot skeleton and intersections"""
        if skel == "skeldist":
            skel = self.st_skeldist[I]
        elif skel == "skeldist_removed":
            skel = self.st_skeldist_removed[I]

        inters = self.st_intersects_dil[I]
        # > Mask
        skel = np.ma.masked_where(skel == 0, skel)
        inters = np.ma.masked_where(inters == 0, inters)

        plt.imshow(skel, cmap="jet", interpolation="none", alpha=1)
        plt.imshow(inters, cmap="jet", interpolation="none", alpha=alpha)


if __name__ == "__main__":
    pass
    # %%

    # > Detailed
    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    kws = dict(
        x_µm=1.5 * 115.4,
    )
    Z = Diameter(
        path=path,
        # denoise=False,
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(method="triangle", sigma=1.5),
        **kws,
    )
    I = 6  # > example
    # Z.mip()

    # %%
    Z

    # %%
    Z.st_segmented.dtype

    # %%
    Z.measure_fiber_width(
        skelet_method="zhang",
        # sigma=1,
        # sigma = None,
        # sampling=.00001,
    )
    # %%
    # plt.imshow(Z.st_skeldist_removed[I], interpolation="none", cmap="jet")
    # %%
    ### Plot overlay of skeleton and segmented image
    Z.plot_masked_by_segmentation(I=I, alpha=0.2)
    skel_rem = Z.st_skeldist_removed[I]
    plt.imshow(skel_rem, interpolation="none", cmap="gist_ncar")
    plt.title("Skeleton with Intersections Removed")

    # %%
    Z.plot_histogram(
        # I=I,
        # bins=50,
        in_µm=True,
    )

    # %%
    hää

    # %%

    # %%
    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, figsize=(16, 8), sharex=True, sharey=True
        )
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title("original")
        ax1.axis("off")
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis("off")
