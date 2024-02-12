"""File to configure all the (sub) caches used in the package"""

# %%
import os

from pprint import pprint

# > Local
from imagep._utils.subcache import SubCache


# %%

# > Location
_os_home_cachedir = os.path.join(os.path.expanduser("~"), ".cache")

# %%
FILTERS = SubCache(
    location=_os_home_cachedir,
    subcache_dir="imagep_filters",
    verbose=True,
    compress=9,
    bytes_limit="3G",  # > 3GB of cache, keeps only the most recent files
)

if __name__ == "__main__":
    pprint(FILTERS.get_cached_inputs())
    