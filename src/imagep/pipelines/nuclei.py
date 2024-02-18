"""Detects nuclei in image and calculates their properties.
- Percentage of area covered by nuclei
"""

# %%

from pprint import pprint

import numpy as np

import matplotlib.pyplot as plt
import imagep._plots.imageplots

# > Local
from imagep.segmentation.segment import Segment


# %%
class Nuclei(Segment):

    def __init__(self, image, **segment_kws):
        super().__init__(image, **segment_kws)

        self.area = None

    #
    # !! == End Class ==================================================


# %%
# == Import ============================================================

if __name__ == "__main__":
    parent = (
        "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/240210 Immunocyto2/"
    )
    paths = [
        parent + "Exp. 1/LTMC Dmp1/",
        parent + "Exp. 2/LTMC Dmp1/",
        parent + "Exp. 3/LTMC Dmp1/",
    ]
    # > contains e.g.: "D0 LTMC DAPI 40x.tif"

    import imagep as ip
    from pathlib import Path

    # > IA_ori = original
    IA_ori = ip.Stack(
        data=paths,
        fname_pattern="*DAPI*.tif",
        # invert=False,
        sort=False,
        imgname_position=[0, 0, 0],  # > Extract a key from the filename
    )
    print(IA_ori.imgs.shape, IA_ori.imgs.dtype)
    _img = IA_ori.imgs[0]
    print(_img.shape, _img.dtype)
    IA_ori.imshow(save_as="0_Original", batch_size=4)  #:: uncomment

    # %%
    IA_ori.imshow()

    # %%
    # ### Too much noise, run median filter
    # from imagep import filters_accelerated
    # Z.imgs = filters_accelerated.median(Z.imgs, size=3)
    # Z.imshow()

    # %%
    ### Preprocess
    I = 2
    # > IA_pp = Image Analysis preprocessed
    IA_pp = ip.PreProcess(
        data=IA_ori,
        median=True,
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(
            # method="triangle",
            # sigma = .7,
            method="otsu",
            sigma=6,
            per_img=True,
        ),
        normalize="per_img",
        remove_empty_slices=False,
        snapshot_index=I,
        # pixel_length=0.1,
    )
    print(IA_pp.info)
    print("per_img: ", IA_pp.background.per_img)
    print("threshold: ", IA_pp.background.threshold)
    # %%
    IA_pp.imgs[0].metadata

    # Zpp.imshow()
    # %%
    IA_pp.plot_snapshots()
    # %%
    # > Show all preprocessed images
    IA_pp.imshow(save_as="1_Preprocessed_All", batch_size=4)
    # %%
    ### Segment
    # TODO: This calls preprocess main
    # Zs = ip.Segment(
    #     data=Zpp,
    #     smoothen_edges_imgs=False,
    #     smoothen_edges_segment=False,
    #     open_segment=False,
    #     segment_method="background",
    # )
    # ip.imshow(Zs.segmented)
    # %%
    ### Quick segmentation
    seg = (IA_pp.imgs > 0).astype(np.uint8)
    print(seg.shape, seg.dtype)
    # ip.imshow(S) # !! Does not preserve image names in plots

    ### Import into an instance of Collection for snapshots and names
    # > IA_seg = Image Analysis segmented
    IA_seg = IA_pp.copy()
    # > Update IA
    IA_seg.imgs = seg
    IA_seg.imshow()  # > preserves image names
    IA_seg.capture_snapshot("Segmentation: Values > 0 are white")

    # %%
    ### Morphology: open (erode + dilate)
    from scipy import ndimage as sp_ndimage

    @ip.preserve_metadata()
    def open(imgs, iterations: int = 5):
        return ip.l2Darrays(
            [
                sp_ndimage.binary_opening(img, iterations=iterations)
                for img in imgs
            ],
            dtype=imgs.dtype,
        )

    segm_opened = open(seg, iterations=5)

    # > Update IA
    IA_seg.imgs = segm_opened.astype(np.float32)
    IA_seg.imshow()
    IA_seg.capture_snapshot("Opening: Erode + Dilate (5 iterations)")

    # %%
    ### Show the full pipeline progress
    IA_seg.plot_snapshots(save_as=f"2_Example_NucleiSegmentation_#{I}")

    # %%
    ### Function for showing contours
    def draw_contours(
        seg,
        masked=False,
    ) -> np.ndarray:
        """Takes binary image mask and returns a contour around that
        mask"""
        ### Get Contours by eroding the mask and subtracting
        seg = seg.astype(np.uint8)
        struct = sp_ndimage.generate_binary_structure(2, 2)
        erode = sp_ndimage.binary_erosion(seg, struct)
        contour = seg ^ erode
        if masked:
            contour = np.ma.masked_where(contour == 0, contour)
        return contour

    ### test
    for img in IA_seg.imgs:
        # contour = draw_contours(img, masked=False)
        contour = draw_contours(img, masked=True)
        print(contour.max())
        # plt.imshow(contour)
        # plt.show()

    # %%
    ### Plot the original images with Segmented Contours
    def plot_seg_contour(
        self: ip.Stack,
        seg,
        batch_size: int = None,
        save_as: str = None,
        ret: bool = False,
    ):
        """Show the difference between two sets of images"""

        ### Init plot
        fig, axes = self.imshow(
            # self,
            max_cols=2,
            scalebar=False,
            share_cmap=True,
            batch_size=batch_size,
            ret=True,
        )

        ### Overlay Contour
        for i, ax in enumerate(axes.flat):
            contour = draw_contours(seg[i])
            ax.imshow(contour, alpha=0.5)

        if save_as is not None:
            imagep._plots.imageplots.savefig(save_as)

        if ret:
            return fig, axes
        else:
            plt.show()

    _ = plot_seg_contour(
        self=IA_pp,
        seg=IA_seg.imgs,
        save_as="3_Segmentmask+Preprocessed",
        batch_size=4,
    )
    _ = plot_seg_contour(
        self=IA_ori,
        seg=IA_seg.imgs,
        save_as="3_Segmentmask+Original",
        batch_size=4,
    )

    # %%
    ### Calculate percentage of area covered by nuclei
    def perc_area_covered(seg: ip.l2Darrays) -> float:
        """Calculates percentage of area covered by nuclei"""

        data = {}
        for i, img in enumerate(seg):
            ### Construct Index
            path = Path(img.folder).parts
            index = (*path, img.name)
            percent = np.sum(img) / np.prod(img.shape)
            # print(np.prod(img.shape), type(np.prod(img.shape)))
            # print(percent, type(percent))
            data[index] = round(percent.item(), 4)
        # > round
        # perc = np.round(perc, 3)
        return data

    # perc = [np.sum(img) / np.prod(img.shape) for img in segm_opened]
    # # > round
    # perc = np.round(perc, 3)
    perc = perc_area_covered(IA_seg.imgs)
    pprint(perc)

    # %%
    # > associate percentages with filekeys
    perc_d = dict(zip(IA_ori.imgnames, perc))
    # > pretty print
    for k, v in perc_d.items():
        print(f"{k}: {v}")
