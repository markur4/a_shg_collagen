"""The main interface to connect EVERYTHING together. This is the main
pipeline, including:
- Importing
- preprocessing
- Analysis
    - segmentation
    - analysis of images or segmented data
- Visualize
    - ZStack
"""
# %%

import numpy as np


# > local
import imagep._utils.utils as ut
from imagep.images.stack import Stack
# from imagep.processing.background import Background

# %%