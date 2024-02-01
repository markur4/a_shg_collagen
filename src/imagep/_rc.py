"""runtime configuration"""
import matplotlib as mpl
# import matplotlib.pyplot as plt

# > local imports
import imagep._utils.utils as ut

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


# == Dtypes ============================================================
