# ImageP: Image Analysis Pipelines in Python.
**Yes, this is a rip-off of ImageJ/Fiji. It's trying to achieve the same
thing (but easier): Scientific image analysis and processing with a
focus on (but not limited to) microscopy.**

## Install
```bash
pip install git+https://github.com/markur4/ImageP
```

## Why ImageP?

#### Differences to ImageJ/Fiji:
- **100% Python**
- **Automation:** Focus on batch processing
- **Straightforward:** 
  - No Plugins, only
  - Active curation of the state-of-the-art tools: *There's 1 way to do things*
- **Comprehensive:** Every processing intermediate is saved and shown so that everyone can understand what's going on
- **Scientific:** Every step is documented and reviewable in the
  `history` attribute
- **Efficient:** Costly computed operations are cached and/or
  parallelized

#### I wrote this because:
- ImageJ/Fiji was always crashing
- I don't need graphical interfaces (but could be added)
- I like batch processing



## Features

### Generalized Pipelines
I think that too much choice is confusing. ***There should be one *best*
way to do things.*** Hence, every pipeline consists of curated
algorithms that just do their job. These pipelines are highly automated
and meant to take care of tedious work, e.g. importing batches of
images.

**Every pipeline consists of these steps:**
1. **Image Import**
2. **Preprocessing**
3. **Segmentation** (skippable)
4. **Analysis and/or Visualization**

You will probably ***interact*** with ImageP via a class from the third or
fourth step of this pipeline. For example, if you want to segment
images, you would import the `Segmentation` class and use it to segment
your images. This is going to run everything before that (importing,
denoising, etc.). If you choose a class from analysis/visualization, it
will inherit and execute functions from segmentation (if it needs that)
using pre-configured parameters.

ImageP still retains ***flexibility*** by allowing every feature to be
switched off and sticking to the universal 3D `numpy` array as the main
data format to store images. That way you can always exit the pipeline
at every step and continue with other tools.

#### 1. Image Import
Import images by automatically detecting the file format. Standard
   precision is set to `np.float32`

#### 2. The PreProcessing-Pipeline
Some things just make everything better. For example, since noise makes
everything worse, we simply remove that first, using an algorithm that
perfectly retains textures. Hence, **every** pipeline inherits features
from the `Preprocessing` class pipeline and starts with those. The goal
is to equalize images to improve *every* kind further analysis. At this
level, we also include basic functionalities ,like adding a scalebar.
The `preprocess` package currently employs these optional processing
steps (in this order):
1. Denoise (non-local means)
2. Subtract Background
3. Normalize pixel values (0-1)
4. Add Scalebar

#### 3. Segmentation
This is necessary for complex analysis pipelines. 
   1. Pronounce features
      - Smoothen edges (median filter)
   2. Train Classifier
      1. Annotate region classes
      2. Train Model (Random-forest)
   3. Segment
      - Background Threshold
      - Random Forest Classification
   4. Postprocess Segmentation
      - Remove small objects
      - Fill holes
      - smoothen edges

#### 4. Analysis / Visualization (100% your choice)
Here's where the "plugins" character from Fiji starts. These are highly
specialized functionalities meant to be expanded in the future. The
`analyse` package contains pipelines to quantify image information e.g.
after segmentation. The current implementation includes:
- Measure Fiber Width
- *More to come...*

The `visualise` package contains tools to visualize complex image data.

- 3D
  - Volume Rendering
  - Animated .gifs
- *More to come...*



### Inheritance Tree

```mermaid

classDiagram
   



   %% == Import ======================================================
   class numpy_ndarray{
      ...
      ....()
   }
   class import {
      ...
      from_textfile()
   }
   class Images{
      pixel_size
      x_µm
      y_µm
      ....()
   }
   
   
   #%% == Preprocessing ==============================================
   
   
   class PreProcess {
      ...
      ....()
   }
   
   #%% == Visualize ===============================================
   class ZStack{
      z_µm
      ....()
    }
   class Annotate{
      ...
      ....()
   }
   class Segment{
      ...
      ....()
   }
   class FibreDiameter{
      data_check_integrity()
      ....()
   }
   

   numpy_ndarray *-- import
   import *-- PreProcess
   PreProcess <-- ZStack
   PreProcess <-- Annotate
   Annotate <-- Segment
   Segment <-- FibreDiameter



```



