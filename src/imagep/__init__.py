#
# == Execute runtime configuration =====================================
from . import _rc

# == Flatten Module Access =============================================
# > The following imports are for direct access to the modules
# > The comments show the convention for correct import statements
# :: 'import imagep as ip'

### Import & Processing Interfaces
# :: 'ip.<Interface>(**kws)'
from .images.imgs import Img, Imgs  # :: ip.Imgs(**kws)
from .processing.preprocess import PreProcess  # :: ip.PreProcess(**kws)
from .segmentation.segment import Segment  # , Annotate


### Pipelines Visualization
# :: 'ip.<Pipeline>(**kws)'
from .visualize.zstack import ZStack  # :: ip.ZStack(**kws)


### Pipelines for Analysis
# :: 'ip.<Pipeline>(**kws)'
from .pipelines.diameter import FibreDiameter  # :: ip.FibreDiameter(**kws)


### Libraries for tools
from .processing import filters  # :: e.g. 'ip.filters.denoise(**kws)'
from .processing import filters_accelerated
from ._utils import utils  # :: 'ip.utils.<function>(**kws)'

# > Plotting tools are imported directly
# :: 'import ip.imageplots as ipl' 
from ._plots.imageplots import * # :: e.g. 'ip.imshow(**kws)'
from ._plots.dataplots import * # :: e.g. 'ip.histogram(**kws)'
from ._plots.scalebar import * # :: e.g. 'ip.burn_scalebars(**kws)'
