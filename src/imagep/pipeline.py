"""The main interface to connect EVERYTHING together. This is the main
pipeline, including:
- Importing
- Preprocessing
- Visualize
    - e.g. ZStack
- Analysis
    - segmentation
    - analysis of images or segmented data
    - e.g. 
    
Principles:
- Most modules are accessed via composition
- The imgs attribute is being being overwritten
- Method chaining (methods return self)
"""

# %%
from collections import OrderedDict
import numpy as np


# > local
import imagep._configs.rc as rc
import imagep.types as T
import imagep._utils.utils as ut
from imagep.images.stack import Stack
from imagep.arrays.l2Darrays import l2Darrays
from imagep.processing.filter_class import Filter
from imagep.processing.background import Background
from imagep.processing.process import Process
from imagep.processing.preprocess import PreProcess

# from imagep.processing.background import Background


# %%
# ======================================================================
# == Class: Pipeline ===================================================
class Pipeline(Process):
    def __init__(
        self,
        ### StackImport kws:
        data: T.source_of_imgs = None,
        verbose: bool = True,
        ### StackMeta kws:
        pixel_length: float | list[float] = None,
        unit: str = "µm",
        scalebar_length: int = None,  # > in (micro)meter
        ### fileimport_kws
        fname_pattern: str = "",
        fname_extension: str = "",
        sort: bool = True,
        imgname_position: int | list[int] = 0,
        invertorder: bool = True,
        dtype: np.dtype = rc.DTYPE,
        ### Process kws
        snapshot_index: int = None,
        keep_original: bool = True,
        ### kws for importfunction, like skiprows
        **importfunc_kws,
    ):
        super().__init__(
            data=data,
            verbose=verbose,
            pixel_length=pixel_length,
            unit=unit,
            scalebar_length=scalebar_length,
            fname_pattern=fname_pattern,
            fname_extension=fname_extension,
            sort=sort,
            imgname_position=imgname_position,
            invertorder=invertorder,
            dtype=dtype,
            snapshot_index=snapshot_index,
            keep_original=keep_original,
            **importfunc_kws,
        )
        ### Initialize components that can be added by composition
        self.snapshots: OrderedDict[str, np.ndarray] = OrderedDict()
        self.background: Background = None
        self.filter: Filter = None

    def preprocess(
        self,
        median: bool = False,
        denoise: bool = True,
        normalize: bool = True,
        subtract_bg: bool = True,
        subtract_bg_kws: dict = dict(method="otsu", sigma=3, per_img=False),
    ) -> Process:
        """Preprocess images"""
        pp = PreProcess(
            data=self,
            median=median,
            denoise=denoise,
            normalize=normalize,
            subtract_bg=subtract_bg,
            subtract_bg_kws=subtract_bg_kws,
        )
        ### Composition
        self.snapshots.update(pp.snapshots)
        self.background = pp.background

        ### Update imgs
        self.imgs = pp.imgs

        return self


if __name__ == "__main__":
    """Here's how I imagine interfacing with class Pipeline to program a
    pipeline
    import

    ppl = ip.Pipeline(
        data = path,
        unit = "µm",
        scalebar_length = 100,
        ** Import kws & Process kws
    )
    ppl = (
        ppl
        .preprocess(
            median = False,
            denoise = True,
            normalize = True,
            subtract_bg = True,
            subtract_bg_kws = dict(),
        )
        .segment(
            method = {"background", "watershed"},
            kws = None
        )
    )
    """

    import imagep as ip

    parent = (
        "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/"
    )
    ### Import from a txt file.
    # > Rough
    path = parent + "1 healthy z-stack rough/"
    z_dist = 10 * 0.250  # > stepsize * 0.250 µm
    # > x_µm = fast axis amplitude * calibration =  1.5 V * 115.4 µm/V
    pixel_length = (1.5 * 115.4) / 1024

    # > Detailed
    # path = parent + "2 healthy z-stack detailed/"
    # z_dist = 2 * 0.250  # > stepsize * 0.250 µm
    # pixel_length = (1.5 * 115.4) / 1024

    I = 8
    PPL = Pipeline(
        data=path,
        fname_extension=".txt",
        imgname_position=1,
        scalebar_length=10,
        snapshot_index=I,
        pixel_length=pixel_length,
    )
    # %%
    PPL = PPL.preprocess(
        median=False,
        denoise=True,
        normalize=True,
        subtract_bg=True,
        subtract_bg_kws=dict(method="otsu", sigma=3, per_img=False),
    )

    # %%
    PPL.plot_snapshots()
