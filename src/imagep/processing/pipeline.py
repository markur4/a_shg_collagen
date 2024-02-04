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
import imagep._rc as rc
import imagep._utils.utils as ut
from imagep.images.imgs import Imgs
from imagep.processing.filter_class import Filter

# from imagep.processing.background import Background
import imagep._plots.imageplots as imageplots


# %%
# == Class Process =====================================================
class Pipeline(Imgs):
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
    def snapshots_array(self) -> np.ndarray:
        """Returns snapshots as an array"""
        return np.array(list(self.snapshots.values()))

    def plot_snapshots(
        self,
        return_fig_axes=False,
        saveto: str = None,
    ) -> None | tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plot sample images from preprocessing steps"""

        ### Check if samples were collected
        if len(self.snapshots) == 0:
            raise ValueError("No snapshots were collected")

        # ### Retrieve steps
        # steps = list(self.snapshots.keys())

        ### Plot
        fig, axes = imageplots.imshow(
            self.snapshots_array,
            max_cols=2,
            scalebar=True,
            scalebar_kws=dict(
                pixel_size=self.pixel_size, microns=self.scalebar_microns
            ),
            share_cmap=False,
        )

        ### Edit ax titles
        for (
            i,
            (ax, (step, img)),
        ) in enumerate(zip(axes.flat, self.snapshots.items())):

            AXTITLE = (
                f"{i}. step: '{step}'\n"
                f"{img.shape[0]}x{img.shape[1]}  {img.dtype}"
            )
            ax.set_title(AXTITLE, fontsize="medium")

        ### Add fig title
        I = self.snapshot_index
        T = self._shape_original[0]
        FIGTITLE = (
            f"Snapshots acquired after each processing step\n"
            f"   Took image #{I+1}/{T} (i={I}/{T-1}) as sample"
        )
        imageplots.figtitle_to_plot(FIGTITLE, fig=fig, axes=axes)
        
        if saveto:
            ut.saveplot(fname=saveto, verbose=self.verbose)
        
        if return_fig_axes:
            return fig, axes

        # axtit = (
        #         f"Image {i+1}/{len(self.imgs)} (i={_i}/{self._num_imgs-1})"
        #         f"    {img.shape[0]}x{img.shape[1]}  {img.dtype}"
        #         # f"\nmin={form(img.min())}  mean={form(img.mean())}  max={form(img.max())}"
        #     )
        #     ax.set_title(axtit, fontsize="medium")

        # ### Fig title
        # tit = f"{self.path_short}\n - {self._num_imgs} Total images"
        # if self._slice:
        #     tit += f"; Sliced to {len(_imgs)} image(s) (i=[{self._slice}])"

        # ### Get number of rows in axes
        # bbox_y = 1.05 if axes.shape[0] <= 2 else 1.01
        # fig.suptitle(tit, ha="left", x=0.01, y=bbox_y, fontsize=12)
