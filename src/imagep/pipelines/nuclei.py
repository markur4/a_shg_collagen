"""Detects nuclei in image and calculates their properties.
- Percentage of area covered by nuclei
"""

# %%
from typing import Callable
from pprint import pprint

import numpy as np

import matplotlib.pyplot as plt
import imagep._plots._plotutils as imageplots

# import imagep._utils.metadata as meta

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
        parent + "Exp. 1/LTMC Sost/",
        parent + "Exp. 2/LTMC Dmp1/",
        parent + "Exp. 2/LTMC Sost/",
        parent + "Exp. 2/2D Dmp1/",
        parent + "Exp. 2/2D Sost/",
        parent + "Exp. 3/LTMC Dmp1/",
        parent + "Exp. 3/LTMC Sost/",
        parent + "Exp. 3/2D Dmp1/",
        parent + "Exp. 3/2D Sost/",
        
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
        # imgname_position=[0, 0, 0],  # > Extract a key from the
        # filename
        imgname_position=0
    )
    print(IA_ori.imgs.shape, IA_ori.imgs.dtype)
    _img = IA_ori.imgs[0]
    print(_img.shape, _img.dtype)

    # %%
    IA_ori.imshow(save_as="0_Original", batch_size=4)  #:: uncomment

    # %%
    # ### Too much noise, run median filter
    # from imagep import filters_accelerated
    # Z.imgs = filters_accelerated.median(Z.imgs, size=3)
    # Z.imshow()

    # %%
    ### Preprocess
    I = 2
    # * IA_pp = Image Analysis preprocessed
    IA_pp = ip.PreProcess(
        data=IA_ori,
        median=True,
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(
            # method="triangle",
            # sigma = .7,
            # sigma = 2,
            method="otsu",
            sigma=3,
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
    # ip.imshow(S) 

    ### Import into an instance of Collection for snapshots and names
    # > IA_seg = Image Analysis segmented
    IA_seg = IA_pp.copy()
    # > Update IA
    IA_seg.imgs = seg
    # IA_seg.imshow()  
    IA_seg.capture_snapshot("Segmentation: Values > 0 are white")
    #%%
    ### Problematic image
    IA_ori[30].imshow()  
    IA_pp[30].imshow()
    IA_seg[30].imshow()  
    # %%
    ### Morphology: 
    # > close (dilate + erode) to remove small holes
    # > open (erode + dilate) to remove small objects
    
    from scipy import ndimage as sp_ndimage

    @ip.preserve_metadata()
    def close(imgs, iterations: int = 5):
        return ip.l2Darrays(
            [
                sp_ndimage.binary_closing(img, iterations=iterations)
                for img in imgs
            ],
            dtype=imgs.dtype,
        )

    @ip.preserve_metadata()
    def open(imgs, iterations: int = 5):
        return ip.l2Darrays(
            [
                sp_ndimage.binary_opening(img, iterations=iterations)
                for img in imgs
            ],
            dtype=imgs.dtype,
        )
    seg = close(seg, iterations=5)
    seg = open(seg, iterations=5)

    # > Update IA
    IA_seg.imgs = seg.astype(np.float32)
    IA_seg[30].imshow()
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
        # print(contour.max())
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
        figs_axes = self.imshow(
            max_cols=2,
            scalebar=False,
            share_cmap=True,
            ret=True,
            batch_size=batch_size,
        )
        for i_batch, (fig, axes) in enumerate(figs_axes):
            ### Overlay Contour
            for i, ax in enumerate(axes.flat):
                index = i_batch * batch_size + i
                contour = draw_contours(seg[index])
                ax.imshow(contour, alpha=0.5)

        imageplots.return_plot_batched(
            figs_axes,
            save_as=save_as,
            ret=ret,
        )

    _ = plot_seg_contour(
        self=IA_pp,
        seg=IA_seg.imgs,
        save_as="3_Segmentmask-Preprocessed",
        batch_size=4,
    )
    # %%
    _ = plot_seg_contour(
        self=IA_ori,
        seg=IA_seg.imgs,
        save_as="3_Segmentmask-Original",
        batch_size=4,
    )

    # %%
    ### Calculate percentage of area covered by nuclei
    def _calc_perc(img: np.ndarray) -> float:
        perc = np.sum(img) / np.prod(img.shape)
        return round(perc.item(), 4)
    
    def calc_and_record(imgs:ip.l2Darrays, op:Callable):
        """Calculates percentage of area covered by nuclei"""

        data = []
        for img in imgs:
            ### Calculate total intensity
            result = op(img)
            
            ### Construct Record
            path = Path(img.folder).parts
            record = (*path, img.name, result)
            data.append(record)

        return data

    perc = calc_and_record(IA_seg.imgs, op=_calc_perc)
    pprint(perc)

    # %%
    ### Make dataframe
    import pandas as pd

    def records_to_df(list_records: list[tuple]) -> pd.DataFrame:
        """Converts dictionary to dataframe. Dictionary has tuple of
        indices as keys and a scalar as value. Indices start with an
        integer row index. Returns a dataframe with indices expanded to
        columns.
        """

        ### Convert the list of records into a DataFrame
        df = pd.DataFrame(
            list_records,
            columns=[
                "folder1",
                "folder2",
                "img_name",
                "result",
            ],
        )

        return df

    df = records_to_df(list_records=perc)
    df
    #%%
    ### save dataframe as xlsx
    df.to_excel("4_perc_nuclei.xlsx", index=False)
