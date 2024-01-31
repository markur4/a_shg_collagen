"""Here are all filters:
- all static and on global namespace
- all have similar interface
- If accelerated, then parallelized and cached
"""
#%%

import numpy as np


import scipy as sp
import skimage as ski

# > local imports
import imagep._utils.utils as ut
from imagep._utils.subcache import SubCache


# %%
# == CACHE =============================================================

