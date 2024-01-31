"""runtime configuration"""

# == matplotlib ========================================================
import matplotlib as mpl
import matplotlib.pyplot as plt


### Image quality
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["image.interpolation"] = "none"

### Plot properties
mpl.rcParams["image.cmap"] = "gist_ncar"
mpl.rcParams["figure.facecolor"] = "darkgrey"
mpl.rcParams["savefig.facecolor"] = "None"

### Font
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial Narrow"]