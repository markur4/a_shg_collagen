"""runtime configuration"""

# == matplotlib ========================================================
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["image.interpolation"] = "none"
mpl.rcParams["image.cmap"] = "gist_ncar"

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial Narrow"]