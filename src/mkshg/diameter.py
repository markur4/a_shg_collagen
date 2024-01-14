#
# %%

from pprint import pprint

from collections import OrderedDict

from pathlib import Path
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

### increase display dpi of matplotlib
plt.rcParams["figure.dpi"] = 300
import seaborn as sns

### Analysis
import scipy as sp

# from scipy import ndimage

import skimage.morphology as morph
from skimage import draw
from skimage import filters

# from skimage import data
# from skimage.util import invert

# from skimage import filters

# > Internal
from mkshg.import_and_preprocess import PreProcess

# %%


class Diameter(PreProcess):
    def __init__(self, use_mip=True, **preprocess_kwargs):
        ### Don't burn a scalebar, otherwise its size will be included
        preprocess_kwargs["scalebar_micrometer"] = False

        super().__init__(**preprocess_kwargs)

        # TODO: Implement Segmentation
        self.segmented = np.zeros(self.stack.shape)

        ### List the stacks for processing
        self.edt = np.zeros(self.segmented.shape)
        self.skeleton = np.zeros(self.segmented.shape)

        self.intersects = np.zeros(self.segmented.shape)
        self.intersects_disks = np.zeros(self.segmented.shape)

        self.skeleton_edt = np.zeros(self.segmented.shape)
        self.skeleton_edt_nointer = np.zeros(self.segmented.shape)

        self.diameters_micro = []
        self.diameters_micro_flat = []
        self.diameters_px = []
        self.diameters_px_flat = []

        ### Execute transform pipeline
        self.measure_fiber_width(use_mip=use_mip)

    def measure_fiber_width(
        self,
        # sigma: float = 1,
        use_mip=True,
        skelet_method="zhang",
        **edt_kws,
    ) -> None:
        """Pipeline for measuring fiber width"""

        ### Perform Analysis on MIP or on each image individually?
        # > Advantages of MIP:
        # - Higher Quality (less noise, stronger signal)
        # - Thresholding on mip is robuster, higher background, ignores
        # ' weak signals
        # - Orientation of fibers now irrelevant
        # - Less computation time
        # > Disadvantages of MIP:
        # - If too many fibers, segmentation becomes impossible

        self.stack_mip = np.array(
            [
                self.mip(
                    show=False,
                    savefig=False,
                    return_array=True,
                )
            ]
        )
        if use_mip:
            self._stack_use = self.stack_mip
        else:
            self._stack_use = self.stack

        ### Start with segmented stack
        self.segmented = self.segmentation(
            self._stack_use, thresh_per_img=True, sigma=4
        )

        ### Distance Transform
        self.edt = self.distance_transform(
            self.segmented,
            **edt_kws,
        )

        ### Skeletonize
        self.skeleton = self.skeletonize(
            # self.st_distance,
            self.segmented,
            method=skelet_method,
        )

        ### Intersections
        # > Detect Intersections
        self.intersects = self.detect_intersections(self.skeleton)
        self.intersects_disks = self.dilate_intersections(
            intersects=self.intersects,
            distance_transform=self.edt,
        )

        ### Apply distances to Skeleton
        self.skeleton_edt = self.skeleton * self.edt
        # > Removing Intersections leaves skeleton with gaps, intensity
        # ' of skeleton is half the diameter
        self.skeleton_edt_nointer = self.remove_intersections(
            skeleton=self.skeleton_edt,
            intersects_disks=self.intersects_disks,
        )

        ### Calculate Diameters
        self.diameters_px = self.calc_diameters(
            skeldist=self.skeleton_edt_nointer
        )
        self.diameters_px_flat = np.concatenate(self.diameters_px)

        self.diameters_micro = self.calc_diameters(
            skeldist=self.skeleton_edt_nointer, in_µm=True
        )
        self.diameters_micro_flat = np.concatenate(self.diameters_micro)

    #
    # == Pipeline =========================================================

    @staticmethod
    def segmentation(
        S: np.ndarray,
        thresh_per_img=True,
        sigma=None,
    ) -> np.ndarray:
        """Arbitrary segmentation, makes binary mask: replaces
        background with 0 and foreground with 1"""

        ### Init empty stack
        stack = np.zeros(S.shape)

        ### Segmentation
        for i, img in enumerate(S):
            ### Recalculate threshold for each individual image
            if thresh_per_img:
                img_blur = filters.gaussian(img, sigma=sigma)
                threshold = filters.threshold_triangle(img_blur)
            else:
                threshold = 0

            ### Segment
            stack[i] = img > threshold

        ### Improve quality of Segmentation
        for i, img in enumerate(stack):
            # > Closing + opening to remove small holes and salt/pepper
            # > Closing = dilation, then erosion
            # > Opening = erosion, then dilation
            # stack[i] = morph.closing(img)
            stack[i] = morph.opening(stack[i])
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
    def skeletonize(S: np.ndarray, method="zhang") -> np.ndarray:
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
        skeleton: np.ndarray, intersects_disks: np.ndarray
    ) -> np.ndarray:
        """Removes the dilated intersections from the skeleton.
        Removing Intersections leaves skeleton with gaps, intensity of
        skeleton is half the diameter"""

        stack = np.zeros(skeleton.shape)  # Init empty stack
        for i, skel in enumerate(skeleton):
            skel_rem = np.where(intersects_disks[i] > 0, 0, skel)
            # > Replace zeros with nan
            skel_rem = np.where(skel_rem == 0, 0, skel_rem)

            stack[i] = skel_rem

        return stack

    def calc_diameters(self, skeldist: np.ndarray, in_µm=False) -> np.ndarray:
        """Calculate the diameter of each fiber in each image"""

        ### Replace zeros with nan
        skeldist = np.where(skeldist == 0, np.nan, skeldist)

        ### Calculate diameters for every image
        diameters = []
        for img in skeldist:
            ### Get all the values of the skeleton
            values = img[~np.isnan(img)] * 2

            ### Convert to µm
            if in_µm:
                values = values * self.pixel_size

            ### Calculate the diameter of each fiber
            diameters.append(values)

        return diameters

    #
    # == Diameter Statistics ===========================================

    def plot_histogram(
        self,
        I: int = None,
        bins: int = 50,
        in_µm=True,
        **kws,
    ):
        """Plot histogram of fiber width"""


        diameters = (
            self.diameters_px_flat if I is None else self.diameters_px[I]
        )

        ### Convert to µm
        if in_μm:
            diameters = diameters * self.pixel_size

        # bins_auto = int(round(max(diameters)))
        # bins = bins_auto if bins == "max" else bins

        ### Plot
        fig, ax = plt.subplots(figsize=(4, 2))
        KWS = dict(
            bins=bins,
            log=True,
        )
        KWS.update(kws)

        ax = plt.hist(diameters, **KWS)

        ### Edit plot
        plt.xlabel("Fiber width [µm]")
        plt.ylabel("Count of Pixels in Skeleton")

    def get_diameters_statistics(self, in_µm: True) -> pd.DataFrame:
        """Retrieves statistics of fiber diameter across image stack
        Rationale:
        - The number of small fibers is HUGELY
        overrepresented, because they are more numerous, since they are
        more likely to be fragmented, and because skeletonization is
        probably more likely to detect smaller fibers than thick fibers.
        - The size of the thickest fiber is good enough, if we're
        interested in level of collagen degradation
        - We require
        corrections for skeleton fragment length and appearance in order
        to get the true distribution of fibre thicknesses in the image
        stack
        """

        ### Initialize dataframe
        Data = pd.DataFrame(
            columns=[
                "max",
                "99.9th percentile",
                "99th percentile",
                "75th percentile",
                "median",
                "min",
            ]
        )

        ### Use µm or pixels as unit?
        if in_µm:
            Diameters = self.diameters_micro
        else:
            Diameters = self.diameters_px

        ### Get statistics for each image
        for diameters in Diameters:
            if len(diameters) > 0:
                d = {}
                ### Get metrics
                d["max"] = np.max(diameters)
                d["99.9th percentile"] = np.quantile(diameters, 0.999)
                d["99th percentile"] = np.quantile(diameters, 0.99)
                d["75th percentile"] = np.quantile(diameters, 0.75)
                d["median"] = np.median(diameters)
                d["min"] = np.min(diameters)

                # > Put into dataframe
                Data = Data.append(d, ignore_index=True)

        return Data

    def plot_diameters_statistics(self, in_µm: True) -> None:
        """Plot statistics of fiber diameter across image stack"""
        Data = self.get_diameters_statistics(in_µm=in_µm)

        ### Plot
        Data.plot()
        plt.title("Diameter Statistics Across Image Stack")
        plt.ylabel("Fiber Diameter [µm]")
        plt.xlabel("Image Number")

    # == Visualization of Intermediates ================================

    def plot_overlay(self, img="stack", I=0, skelet_alpha=1):
        """Plot overlay of skeleton and segmented image"""

        ### Image
        if img == "stack":
            img = self._stack_use[I]
        elif img == "segmented":
            img = self.segmented[I]
        elif img == "distance":
            img = self.edt[I]

        plt.imshow(img, interpolation="none", cmap="gray")

        ### Skeleton
        # > Mask
        skel = self.skeleton_edt[I]
        skel = np.ma.masked_where(skel == 0, skel)

        plt.imshow(
            skel, cmap="spring", interpolation="none", alpha=skelet_alpha
        )

    def plot_masked_by_segmentation(self, I=0, alpha=0.4):
        """Plot an image masked by segmentation"""
        img = self._stack_use[I]
        seg = self.segmented[I]

        # > Mask
        masked = np.ma.masked_where(seg == 0, img)

        plt.imshow(masked, interpolation="none", cmap="gray", alpha=alpha)

    def plot_skeleton_and_intersections(
        self, I=0, alpha=0.4, skel="skeldist_removed"
    ):
        """Plot skeleton and intersections"""
        if skel == "skeldist":
            skel = self.skeleton_edt[I]
        elif skel == "skeldist_removed":
            skel = self.skeleton_edt_nointer[I]

        inters = self.intersects_disks[I]
        # > Mask
        skel = np.ma.masked_where(skel == 0, skel)
        inters = np.ma.masked_where(inters == 0, inters)

        plt.imshow(skel, cmap="jet", interpolation="none", alpha=1)
        plt.imshow(inters, cmap="jet", interpolation="none", alpha=alpha)

    def plot_segmented_and_skeleton(self, I=0, intersects=True) -> None:
        ### Plot overlay of skeleton and segmented image
        self.plot_masked_by_segmentation(I=I, alpha=0.2)

        if intersects:
            skel = self.skeleton_edt[I]
        else:
            skel = self.skeleton_edt_nointer[I]

        ### Burn Scalebar into image
        skel = self._add_scalebar_to_img(img=skel, µm=10, thickness_μm=3)

        ### Mask
        skel = np.ma.masked_where(skel == 0, skel)

        ### Convert to µm
        skel = skel * self.pixel_size * 2

        ### Plot skeleton
        sk = plt.imshow(skel, interpolation="none", cmap="gist_ncar")
        if intersects:
            plt.title("Skeleton with Intersections")
        else:
            plt.title("Skeleton with Intersections Removed")

        ### Colorbar
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Diameter [µm]", rotation=270)

        ### Annotate Scalebar length
        self.annotate_barsize(μm=10, thickness_µm=3, color="black")


#
#
# ======================================================================
# == Main ==============================================================

if __name__ == "__main__":
    pass
    # %%
    ### Define scaling
    kws = dict(
        x_µm=1.5 * 115.4,
    )
    # %%
    # > Detailed
    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    # > Rough
    # path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    Z = Diameter(
        path=path,
        use_mip=True,
        # use_mip=False,
        denoise=True,
        # denoise=True,  #:: Denoising Improves Segmentation!
        subtract_bg=True,
        subtract_bg_kws=dict(method="triangle", sigma=1.5),
        # scalebar_micrometer=10,
        **kws,
    )
    I = 6  # > example
    I = 0  # > for mip
    # Z.mip()

    # %%
    Z

    # %%
    plt.imshow(Z.stack_mip[0], interpolation="none", cmap="gray")

    # %%
    Z.skeleton_edt_nointer[I]

    # %%
    skel_rem = Z.skeleton_edt_nointer[I]

    ### Replace nan with 0
    # skel_rem = np.where(np.isnan(skel_rem), 0, skel_rem)
    # print(skel_rem.max())

    # %%

    Z.plot_segmented_and_skeleton(intersects=True)

    # %%
    Z.plot_segmented_and_skeleton(intersects=False)

    # %%
    Z.plot_histogram(
        # I=I,
        # bins=50,
        in_µm=True,
    )

    # %%

    Data = Z.get_diameters_statistics(in_µm=True)

    # %%
    Z.plot_diameters_statistics(in_µm=True)

    # %%
    def to_long_format(self: Diameter) -> pd.DataFrame:
        """Converts the list of lists of diameters into a long format
        dataframe"""
        d = {}
        for i, diameters in enumerate(self.diameters_micro):
            if len(diameters) > 0:
                d[i] = pd.Series(diameters)
        Data = pd.concat(d, names=["img"], axis=0)
        Data = pd.DataFrame(Data, columns=["diameter"])
        Data.reset_index(inplace=True)
        return Data

    Data = to_long_format(Z)
    # import seaborn as sns

    # print(len(Data))
    # sns.boxplot(data=Data, y="diameter", x="img", showfliers=False)

    # %%
    ### Repeat with other data Rough
    # > Rough
    def measure_Diameter(path, i=0, **kws):
        Z = Diameter(
            path=path,
            use_mip=False,
            denoise=True,
            subtract_bg=True,
            scalebar_micrometer=10,
            **kws,
        )

        print("\n")
        print("======================================================")
        print(path)

        o = "diameter2"

        I = 6

        ### Make folder o if not present
        Path(o).mkdir(parents=True, exist_ok=True)

        ### MIP
        Z.mip(savefig=Path(o, f"d{i}_1_mip.png"))
        plt.show()

        ### Example image from Stack
        plt.imshow(Z[I], interpolation="none", cmap="gist_ncar")
        plt.savefig(Path(o, f"d{i}_2_example.png"), dpi=300)
        plt.show()

        ### Disply Skeleton
        Z.plot_segmented_and_skeleton(I=I, intersects=True)
        plt.savefig(Path(o, f"d{i}_3_skeleton_intersects.png"), dpi=300)
        plt.show()

        Z.plot_segmented_and_skeleton(I=I, intersects=False)
        plt.savefig(Path(o, f"d{i}_4_skeleton_no_intersects.png"), dpi=300)
        plt.show()

        ### Histogram
        Z.plot_histogram(
            # I=I,
            # bins=50,
            in_µm=True,
        )
        plt.savefig(Path(o, f"d{i}_5_histogram.png"), dpi=300)
        plt.show()

        ### Statistics
        Z.plot_diameters_statistics(in_µm=True)
        plt.savefig(Path(o, f"d{i}_6_statistics.png"), dpi=300)
        plt.show()

        return Z

    # %%

    paths = [
        "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/",
        "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/",
        "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/3 healthy z-stack rough 2",
    ]

    # for i, path in enumerate(paths):
        # Z = measure_Diameter(path, i=i, **kws)
