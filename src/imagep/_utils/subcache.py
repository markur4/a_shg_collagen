#
# %% Imports


from typing import Callable, List

import os
from pathlib import Path

# from icecream import ic

from joblib import Memory

# from plotastic.utils import utils as ut


class SubCache(Memory):
    """Expands the joblib.Memory class with some useful methods.
    -
    - List directories within cache
    - List objects within cache
    - Adds subcache attribute, with benefits:
        - Subcache replaces module name in cache directory
        - More control over cache directories
        - Persistent caching, since IPythond passes a new location to
          joblib each time the Memory object is initialized
    - Doesn't work right if two SubCache Objects point cache the same function
    """

    def __init__(
        self,
        subcache_dir: str,
        assert_parent: str = None,
        bytes_limit: int | str = "3G",  # > 3GB
        compress: int = 9,
        *args,
        **kwargs,
    ):
        ### Handle Kwargs from joblib.Memory
        KWS = dict(
            compress=compress,
        )
        KWS.update(kwargs)

        ### Inherit from joblib.Memory
        super().__init__(*args, **KWS)

        ### Set maximum size of cache, keeps most recent files
        self.reduce_size(bytes_limit)

        ### Subfolder of location, overrides default subfolder by joblib
        self.subcache_dir = subcache_dir

        ### self.location/joblib/subcache
        self.subcache_path = os.path.join(
            self.location, "joblib", self.subcache_dir
        )

        ### Prevent joblib folders being created by wrong Interactive Windows
        if not assert_parent is None:
            parent_full = Path(self.location).absolute()
            parent = os.path.split(parent_full)[-1]
            assert (
                parent == assert_parent
            ), f"When Initializing joblib.Memory, we expected cache to be in {assert_parent}, but we ended up in {parent_full}"

    def __iter__(self):
        """Iterate over the list of cache directories"""
        for item in self.store_backend.get_items():
            path_to_item = os.path.relpath(
                item.path,
                start=self.store_backend.location,
            )
            yield os.path.split(path_to_item)

    def list_dirs(
        self, detailed: bool = False, max_depth: int = 3
    ) -> List[str]:
        """
        Reads the the cached files and displays the directories

        :param detailed: if True, returns all cache directories with
            full paths. Default is False.
        :type detailed: bool, optional
        :param max_depth: The maximum depth to search for cache
            directories. Default is 4.
        :type max_depth: int, optional
        :return: List[str], a list of cache directories.
        """

        subcache = self.subcache_path

        location_subdirs = []

        ### Recursive walking
        for root, dirs, _files in os.walk(subcache):
            #' Don't go too deep: 'joblib/plotastic/example_data/load_dataset/load_dataset',
            depth = root[len(subcache) :].count(os.sep)
            if not detailed and depth > max_depth:
                continue
            for dir in dirs:
                #' Don't need to check for 'joblib' because it's not a subdirectory of cache_dir
                #' Exclude subdirectories like "c1589ea5535064b588b2f6922e898473"
                if len(dir) >= 32 or dir == "joblib":
                    continue
                #' Return every path completely
                if detailed:
                    location_subdirs.append(os.path.join(root, dir))
                else:
                    dir_path = os.path.join(root, dir)
                    dir_path = dir_path.replace(subcache, "")
                    if dir_path.startswith("/"):
                        dir_path = dir_path[1:]
                    location_subdirs.append(dir_path)
        return location_subdirs

    def get_cached_inputs(self) -> dict[str, dict[str, object]]:
        """Reads the cached files and returns a dictionary with function
        as keys and their inputs as values"""

        inputs = dict()

        for path_to_item in self:
            ### Get the name of the function
            func = os.path.split(path_to_item[0])[-1]

            ### Get input arguments
            args: dict = self.store_backend.get_metadata(path_to_item).get(
                "input_args"
            )

            ### Shorten the inputs if they are too long
            for arg, value in args.items():
                m = "(" if isinstance(value, str) else f"({type(value)};  "
                try:
                    if len(value) > 50:
                        args[arg] = m + f"{value[:25]} ...; length: {len(value)})"
                except TypeError: #> if len() is not applicable
                    if hasattr(value, "shape"):
                        args[arg] = m + f"{value[:25]} ...; shape: {value.shape})"
                        
            ### Store the inputs
            inputs[func] = args
        return inputs

    # def check_entry(self, func: Callable, mem_kwargs: dict) -> bool:
    #     """Check if these arguments are cached"""

    #     ### Get the name of the function
    #     func_name = func.__name__

    #     if not func_name in self.kwargs:
    #         return False
    #     else:
    #         ### try to load the result with the given kwargs
    #         try:
    #             self.cache.load_item((func_name, mem_kwargs))
    #             return True
    #         except KeyError:
    #             return False

    def get_cached_outputs(self) -> dict[str, tuple[dict, object]]:
        """Reads the cached files and returns a dictionary with function
        as keys and as values a tuple of the inputs and outputs of that
        function"""

        entries = dict()

        for path_to_item in self:
            ### Get the name of the function
            func = os.path.split(path_to_item[0])[-1]

            ### Get the result object
            try:
                result = self.store_backend.load_item(path_to_item)
            except KeyError:
                continue

            ### Get input arguments
            args = self.store_backend.get_metadata(path_to_item).get(
                "input_args"
            )

            entries[func] = (args, result)
        return entries

    def list_kwargs(self):
        """Return list of all stored kwargs"""

        kwargs = []
        for kwarg, obj in self.get_cached_outputs():
            kwargs.append(kwarg)
        return kwargs

    def subcache(self, f: Callable, **memory_kwargs) -> Callable:
        """Cache it in a persistent manner, since Ipython passes a new
        location to joblib each time the Memory object is initialized
        """
        f.__module__ = self.subcache_dir
        f.__qualname__ = f.__name__

        return self.cache(f, **memory_kwargs)


# ======================================================================

if __name__ == "__main__":
    pass
    # %%
    ### Define Home directory
    home = os.path.join(
        os.path.expanduser("~"),
        ".cache",
    )
    print(home)

    ### Define this location as cache
    here = os.path.join(".", ".cache")

    ### A test funciton to test caching
    def sleep(seconds):
        import time

        time.sleep(seconds)

    # %%
    # == Cache HOME ====================================================
    MEM = SubCache(location=home, subcache_dir="subcachetest", verbose=True)

    sleep = MEM.subcache(sleep)
    # %%
    sleep(1.4)  # > First time slow, next time fast
    # %%
    MEM.list_dirs(detailed=True)
    # %%
    MEM.get_cached_outputs()

    # %%
    # == Cache HERE =====================================================
    MEM2 = SubCache(location=here, subcache_dir="subcachetest", verbose=True)

    sleep = MEM2.subcache(sleep)
    # %%
    MEM2.list_dirs(detailed=True)
    # %%
    MEM2.get_cached_outputs()

    # %%
    sleep(1.5)  # > First time slow, next time fast

    # %%
    MEM2.clear()

    # %%
    ### Using different cache allows clearance of only that cache
    MEM2 = SubCache(location=home, subcache_dir="plotic2", verbose=True)

    def slep(seconds):
        import time

        time.sleep(seconds)

    sleep_cached2 = MEM2.subcache(slep)
    sleep_cached2(1.4)
    # %%
    MEM2.list_dirs()
    # %%
    MEM2.clear()
