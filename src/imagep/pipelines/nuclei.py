"""Detects nuclei in image and calculates their properties.
- Percentage of area covered by nuclei
"""

# %%

import numpy as np

import matplotlib.pyplot as plt

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
    parent = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/240201 Imunocyto/"
    path = [
        parent + "Exp. 1/Dmp1/",
        parent + "Exp. 2/Dmp1/",
        parent + "Exp. 3 (im Paper)/Dmp1"
    ]
    # > contains e.g.: "D0 LTMC DAPI 40x.tif"

    import imagep as ip
    from pathlib import Path

    Z = ip.Imgs(
        data=path,
        fname_pattern="*DAPI*.tif",
        # invert=False,
        sort=False,
        imgkey_positions=[0, 2, 2], #> Extract a key from the filename
    )
    print(Z.imgs.shape, Z.imgs.dtype)
    _img = Z.imgs[0]
    print(_img.shape, _img.dtype)
    Z.imshow(saveto="Original_all")

    # %%
    # ### Too much noise, run median filter
    # from imagep import filters_accelerated
    # Z.imgs = filters_accelerated.median(Z.imgs, size=3)
    # Z.imshow()

    # %%
    I = 2
    Zpp = ip.PreProcess(
        data=Z,
        median=True,
        denoise=True,
        subtract_bg=True,
        subtract_bg_kws=dict(
            method="triangle",
            sigma=0.7,
            per_img=False,
        ),
        remove_empty_slices=False,
        snapshot_index=I,
    )
    print(Zpp.info)
    print("per_img: ", Zpp.background.per_img)
    print("threshold: ", Zpp.background.threshold)
    # Zpp.imshow()
    Zpp.plot_snapshots(saveto="Preprocessed_D7_snapshots")
    # %%
    # > Show all preprocessed images
    Zpp.imshow(saveto="Preprocessed_All")
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
    S = (Zpp.imgs > 0).astype(np.uint8)
    print(S.shape, S.dtype)
    ip.imshow(S)
    # %%
    ### Morphology: open (erode + dilate)
    import scipy as sp

    S_opened = np.array(
        [sp.ndimage.binary_opening(img, iterations=8) for img in S]
    )
    ip.imshow(S_opened.astype(np.float32), saveto="Segmented_all_opened")

    # %%
    ### Show the original images only where segmented
    S_imgs = Zpp.imgs * S_opened
    ip.imshow(S_imgs[I], saveto="Segmented_D7")
    plt.show()
    print("original")
    Zpp[I].imshow()
    # %%
    ### show difference between original and segmented
    ip.imshow(S_imgs[I] - Zpp.imgs[I], saveto="Segmented_D7_diff")

    # %%
    ### Calculate percentage of area covered by nuclei
    perc = [np.sum(img) / np.prod(img.shape) for img in S_opened]
    # > round
    perc = np.round(perc, 3)

    # > associate percentages with filekeys
    perc_d = dict(zip(Z.imgkeys, perc))
    # > pretty print
    for k, v in perc_d.items():
        print(f"{k}: {v}")
