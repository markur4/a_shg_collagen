"""A file to store scientific plots like histograms, boxplots gained
from image processing"""

# %%

from pathlib import Path

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# > Local
import imagep._rc as rc
import imagep._utils.utils as ut
from imagep._plots.imageplots import imshow


# %%
# == Testdata ==========================================================
if __name__ == "__main__":
    # !! Import must not be global, or circular import
    from imagep.images.imgs import Imgs

    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = Imgs(data=path, verbose=True, x_µm=1.5 * 115.4)
    print("pixelsize=", Z.pixel_length)
    I = 6


# %%
def histogram(array: np.ndarray, bins=100, log: bool = True, cmap="gist_ncar"):

    pixels = array.flatten()

    ### Plot histogram
    # > bars is a BarContainer object
    bars = plt.hist(pixels, bins=bins, log=log)
    axes = plt.gca()
    fig = plt.gcf()

    ### Change colors of bars according to colormap
    # > Make an array of colors from the colormap
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(bars[0])))
    # > Change the color of the bars
    for bar, color in zip(bars[2], colors):
        bar.set_color(color)

    ### Edit the size of the figure
    fig.set_size_inches(4, 1.5)

    ### Remove the facecolor of the axes
    axes.set_facecolor("none")

    ### Plot percentiles lines into the histogram
    p = np.percentile(pixels, [99, 75, 50])
    min, max = np.min(pixels), np.max(pixels)

    ### Plot percentiles as vertical lines into histogram
    labels = [
        "max",
        "99th",
        "75th",
        "median",
        "min",
    ]
    colors = [
        "red",
        "white",
        "white",
        "white",
        "red",
    ]
    linestyles = [
        "solid",
        (0, [1, 2]),
        (0, [1, 1]),
        "solid",
        "solid",
    ]

    for perc, ls, c, lab in zip([min, *p, max], linestyles, colors, labels):
        ### fill the gaps between lines
        plt.axvline(perc, color="black", ls="solid", linewidth=1)
        plt.axvline(perc, color=c, ls=ls, linewidth=1, label=lab)

    ### Add legend
    legend = fig.legend(
        title="Percentiles",
        loc="center right",
        fontsize=10,
        # frameon=False,
        framealpha=0.2,
        bbox_to_anchor=(1.2, 0.5),
    )
    frame = legend.get_frame()
    frame.set_facecolor("black")

    plt.show()


if __name__ == "__main__":
    # I = I + 7
    _ = imshow(Z.imgs[I])
    plt.show()
    p = histogram(Z.imgs[I], bins=100, log=True)
