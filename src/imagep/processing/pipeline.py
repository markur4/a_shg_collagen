"""Base class for processing images
- Block super()__init__ if to avoid re-loading images
- Track and display sample images before and during processing
- Track and display history of processing steps
"""

# %%

from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt

# > local
import imagep._configs.rc as rc
import imagep._plots._plotutils
from imagep.images.l2Darrays import l2Darrays
from imagep.images.stack import Stack
from imagep.processing.filter_class import Filter

# from imagep.processing.background import Background
import imagep._plots.imageplots as imageplots


# %%
# == Class Process =====================================================
class Pipeline(Stack):
    """Base class for processing images
    - Access to transforms and filters
    - Track and display sample images before and during processing
    - Track and display history of processing steps
    """

    def __init__(
        self,
        *imgs_args,
        snapshot_index: int = None,
        keep_original: bool = True,
        **imgs_kws,
    ):
        """Initialize the Process class
        :param imgs_args: Positional arguments for Imgs
        :type imgs_args: list
        :param snapshot_index: Index of the image to be used for
            snapshots to document the processing steps, defaults to 6
        :type snapshot_index: int, optional
        """
        super().__init__(*imgs_args, **imgs_kws)

        ### Collect Samples for each processing step
        self.snapshot_index = (
            self._half_index() if snapshot_index is None else snapshot_index
        )
        self.snapshots: OrderedDict[str, np.ndarray] = OrderedDict()

        ### Keep original images for the user to compare results
        if keep_original:
            # !! Never reference to this in code, because it might not be there
            self.imgs_original = self.imgs.copy()

        ### Update information about the image
        # > e.g. if images are removed, we want to know that
        self._shape_changed = False  # > will change when shape changed
        # todo: set this in remove_empty_images()

    #
    # == Access to filter functions as methods =========================
    @property
    def filter(self):
        return Filter(imgs=self.imgs, verbose=self.verbose)

    #
    # == Snapshots Documenting Process =================================
    def _half_index(self) -> int:
        """Get the index of the image to be used for snapshots"""
        return len(self.imgs) // 2

    def capture_snapshot(self, step: str) -> None:
        """Capture a snapshot of the images during processing"""
        self.snapshots[step] = self.imgs[self.snapshot_index].copy()

    @property
    def snapshots_as_nparray(self) -> np.ndarray:
        """Returns snapshots as an array"""
        return np.array(list(self.snapshots.values()))
    
    @property
    def snapshots_as_l2Darrays(self) -> list:
        """Returns snapshots as l2Darrays"""
        return l2Darrays(list(self.snapshots.values()))

    def plot_snapshots(
        self,
        ret=False,
        save_as: str = None,
    ) -> None | tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plot sample images from preprocessing steps"""

        ### Check if samples were collected
        if len(self.snapshots) == 0:
            raise ValueError("No snapshots were collected")

        # ### Retrieve steps
        # steps = list(self.snapshots.keys())

        ### Plot
        scalebar = False if self.scalebar_length is None else True
        fig, axes = imageplots.imshow(
            self.snapshots_as_l2Darrays,
            max_cols=2,
            scalebar=scalebar,
            scalebar_kws=dict(
                length=self.scalebar_length,
            ),
            share_cmap=False,
        )

        ### Edit ax titles
        for i_step, (ax, (step, img)) in enumerate(
            zip(axes.flat, self.snapshots.items())
        ):
            axtitle = (
                f"{i_step}. step: '{step}'\n"
                f"{img.shape[0]}x{img.shape[1]}  {img.dtype}"
            )
            ax.set_title(axtitle, fontsize="medium")

        ### Add fig title
        i_snap = self.snapshot_index
        i_total = self._shape_original[0]
        figtitle = (
            f"Snapshots acquired after each processing step\n"
            f"   Took image #{i_snap+1}/{i_total} (i={i_snap}/{i_total-1})"
            " as sample"
        )
        imagep._plots._plotutils.figtitle_to_fig(title=figtitle, fig=fig, axes=axes)

        ### Return
        return imagep._plots._plotutils.return_plot(
            fig=fig,
            axes=axes,
            save_as=save_as,
            ret=ret,
            verbose=self.verbose,
        )
