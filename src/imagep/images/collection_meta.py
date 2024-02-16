"""Classes for adding information to both numpy arrays and the stack of
arrays
"""

# %%
from typing import Self
from pathlib import Path

from pprint import pprint

import numpy as np

# > Local
import imagep._utils.utils as ut
import imagep._configs.rc as rc
from imagep.images.collection_import import CollectionImport
from imagep.images.list2Darrays import list2Darrays
from imagep.images.mdarray import mdarray

# import imagep.images.importtools as importtools


# %%
# == Extract and apply metadata =======================================
def extract_metadata(larry: list2Darrays) -> list[dict]:
    """Extracts metadata from images and returns a list of
    dictionaries with key and value pairs"""
    metadata = [img.metadata for img in larry]
    return metadata


def apply_metadata(
    larry: list2Darrays,
    metadata: list[dict],
) -> None:
    """Applies metadata to images"""
    for img, md in zip(larry, metadata):
        for k, v in md.items():
            setattr(img, k, v)


# %%
# ======================================================================
# == Class CollectionMeta ============================================
class CollectionMeta(CollectionImport):
    """Class for assigning more metadata to stacks of images"""

    def __init__(
        self,
        ### ImgsImport kws:
        data: str | Path | np.ndarray | list[np.ndarray] | Self = None,
        verbose: bool = True,
        ### Metadata:
        pixel_length: float | list[float] = None,
        unit: str | list[str] = rc.UNIT_LENGTH,
        scalebar_length: int = None,  # > in (micro)meter
        ### ImgsImport kws
        **fileimport_kws,
    ):
        ### Inherit
        super().__init__(data, verbose, **fileimport_kws)

        ### Metadata
        # > Collect Metadata
        self._metadata_perfolder = dict(
            pixel_length=pixel_length,
            unit=unit,
        )
        # > Check and rearrange passed metadata
        self.metadata = self._init_metadata()
        # > Add metadata to images according to folders
        self._add_further_metadata_per_folders()

        ### Metadata for the complete stack (not folders)
        self.scalebar_length = scalebar_length

    #
    # == Assign metadata to folder(s) ==================================

    def _check_metadata(self):
        """Checks if metadata is compatible with the object:
        - Values must have same length as folders, or be a single value
        - If single value, it's duplicated for each folder
        """

        ### shorten access
        _MD_RAW = self._metadata_perfolder

        ### Remove None or undefined values
        _MD_RAW = {k: v for k, v in _MD_RAW.items() if not v is None}

        ### Check if metadata values are ...
        # - a single value
        # - or a list with the same length as the number of folders
        kws = dict(target_key="folders", target_n=len(self.path_short))
        for k, v in _MD_RAW.items():
            _MD_RAW[k] = ut.check_samelength_or_number(key=k, val=v, **kws)

        return _MD_RAW

    def _init_metadata(self) -> dict:
        """Makes a dict with shortpaths as keys and metadata as values.
        each metadata is a dict with metadata keys as keys and values as
        values."""

        # > Message
        if self.verbose:
            print("=> Checking & assigning metadata to each folder ...")

        ### Process metadata defined by arguments
        # > Check metadata and expand to full length
        _MD_raw = self._check_metadata()

        ### Fill in metadata into MD
        # > Initialize outer dict: shortpath -> metadata
        MD = {shortpath: {} for shortpath in self.path_short}
        MD["unknown folder"] = {}  # > Capture images that aren't named
        for k, v in _MD_raw.items():
            for i, val in enumerate(v):
                MD[self.path_short[i]][k] = val

        ### Fill in image names found in every folder
        for img in self.imgs:
            # if img.folder == "unknown folder":
            # raise ValueError(f"Unknown folder for image '{img.name}'")
            MD[img.folder]["imgnames"] = [img.name for img in self.imgs]
        # for folder, names in self.imgnames.items():
        #     MD[folder]["imgnames"] = names

        # > Message
        if self.verbose:
            print("   DONE. See Metadata:")
            pprint(MD, sort_dicts=False, compact=True)
            print()

        return MD

    def _add_further_metadata_per_folders(self):
        """Adds metadata defined in constructor to images per folder"""

        for i, img in enumerate(self.imgs):
            # > Get metadata for folder
            if img.folder == "unknown folder":
                # print(f"Unknown folder for image '{img.name}'. Skipping ...")
                continue
            metadata = self.metadata[img.folder]
            for mdkey, mdval in metadata.items():
                setattr(self.imgs[i], mdkey, mdval)

    #
    # == Extract metadata ==============================================

    def extract_metadata(self) -> list[dict]:
        """Extracts metadata from images and returns a list of
        dictionaries with key and value pairs"""
        return extract_metadata(self.imgs)

    def apply_metadata(self, metadata: list[dict]) -> None:
        """Applies metadata to images"""
        return apply_metadata(self.imgs, metadata)

    # !! == End Class ==================================================


if __name__ == "__main__":
    parent = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/240201 Imunocyto/"
    folders = [
        parent + "Exp. 1/Dmp1/",
        parent + "Exp. 2/Dmp1/",
        parent + "Exp. 3 (im Paper)/Dmp1",
    ]
    # folders = parent + "Exp. 1/Dmp1/"
    # > contains e.g.: "D0 LTMC DAPI 40x.tif"

    Z = CollectionMeta(
        data=folders,
        fname_pattern="*DAPI*.tif",
        # invert=False,
        sort=False,
        imgname_position=[
            0,
            2,
            2,
        ],  # > Extract a key from the filename
        pixel_length=[
            0.1,
            0.2,
            0.3,
        ],
        # unit="µm",
        # [
        #     "µm",
        #     # "µm",
        #     # "µm",
        # ],
    )
    pprint(Z.metadata)

    # %%
    Z.imgs_dict

    # %%
    ### Check metadata of imgs from import
    print(Z.imgs[0].name)
    print(Z.imgs[0].folder)

    # %%
    print(Z.imgs[0]._array_str)
    print(Z.imgs[3]._array_str)
    print(Z.imgs[4]._array_str)
