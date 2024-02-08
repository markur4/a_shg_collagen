"""runtime configuration"""

# %%
from typing import TypedDict, Callable

from pathlib import Path

import matplotlib as mpl

# import matplotlib.pyplot as plt

import numpy as np

# > local imports
import imagep._utils.utils as ut

#
# == Img Properties ====================================================
UNIT_LENGTH = "µm"

#
# == Dtypes ============================================================
DTYPE: np.dtype = np.float32

#
# == Importing from files ==============================================

IMPORTCONFIG = ut.ConfigImport(
    # fname_extension=".txt",
    sort=True,
    # sortkey=lambda x: int(Path(x).stem.split("_")[-1]),
    invertorder=True,
    dtype=DTYPE,
    # import_kws=dict(),
)
# print(IMPORTCONFIG)
# IMPORTCONFIG.update(dict(fname_extension=".png", sortkey= lambda x: x*2))
# print(IMPORTCONFIG)

#
# == matplotlib ========================================================

### Image quality
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["image.interpolation"] = "none"

### Plot properties
mpl.rcParams["image.cmap"] = "gist_ncar"
mpl.rcParams["figure.facecolor"] = "darkgrey"
mpl.rcParams["savefig.facecolor"] = "None"

mpl.rcParams["legend.fancybox"] = False

### Font
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial Narrow"]

#
# == Cache =============================================================

### Total cache size
# > keeps only the most recent files
# > One array of shape (20,1024,1024) in float 32 takes 40 MB
CACHE_LIMIT: str = "5G"  # > 5 GB of cache

### Core utilization
# > Numbers of cores in percent of total cores
# > 33% is 3 of 8 cores!
CORES_HALF: float = ut.cores_from_percent(0.50)
CORES_75: float = ut.cores_from_percent(0.75)
CORES_BUT_ONE: float = ut.cores_from_percent(1.00) - 1


# == Debugging =========================================================
DEBUG = False

_EXPONENT = 2
