"""Base class for processing images
- Block super()__init__ if to avoid re-loading images
- Track and display sample images before and during processing
- Track and display history of processing steps
"""
#%%

import numpy as np

import matplotlib.pyplot as plt

# > local
import imagep._rc as rc
import imagep._utils.utils as ut
from imagep.images.imgs import Imgs
from imagep.processing.transforms import Transform


#%%
# == Class Process =====================================================
class Process(Imgs):
    """Base class for processing images
    - Block super()__init__ if to avoid re-loading images
    - Track and display sample images before and during processing
    - Track and display history of processing steps
    """

    def __init__(
        self,
        *imgs_args,
        **imgs_kws,
    ):
        super().__init__(*imgs_args, **imgs_kws)

        ### Collect Samples for each processing step
        self.history_imgs = dict()

    #
    # == Access to transform methods ===================================
    @property
    def transform(self):
        return Transform(imgs=self.imgs, verbose=self.verbose)

    #
    # == Sample Images =================================================
    def plot_process_history(self):
        """Plot sample images from preprocessing steps"""
        ### Check if samples were collected
        if len(self.history_imgs) == 0:
            print("No samples collected")
            return

        ### Plot
        fig, axs = plt.subplots(
            1,
            len(self.history_imgs),
            figsize=(len(self.history_imgs) * 5, 5),
        )
        for i, (k, v) in enumerate(self.history_imgs.items()):
            axs[i].imshow(v)
            axs[i].set_title(k)
            axs[i].axis("off")