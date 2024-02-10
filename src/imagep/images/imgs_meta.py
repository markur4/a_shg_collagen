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
import imagep._rc as rc
from imagep.images.imgs_import import ImgsImport
# import imagep.images.importtools as importtools


# %%
# ======================================================================
# == Class ImgsWithMetadata ============================================
class ImgsMeta(ImgsImport):
    """Class for assigning more metadata to stacks of images"""

    def __init__(
        self,
        ### ImgsImport kws:
        data: str | Path | np.ndarray | list[np.ndarray] | Self = None,
        verbose: bool = True,
        ### Metadata
        pixel_length: float | list[float] = None,
        unit: str | list[str] = rc.UNIT_LENGTH,
        scalebar_length: int = None,  # > in (micro)meter
        ### ImgsImport kws
        **fileimport_kws,
    ):
        ### Inherit
        super().__init__(data, verbose, **fileimport_kws)

        ### Metadata
        # > Metadata for (individual) images
        self._metadata_perfolder = dict(
            pixel_length=pixel_length,
            unit=unit,
        )
        # > Message
        if self.verbose:
            print("=> Checking & Assigning metadata")
        # > Check metadata and rearrange to folders
        self.metadata = self._init_metadata()
        # > Add metadata to images according to folders
        self._add_metadata_per_folders()
        # > Message
        if self.verbose:
            print("  DONE Metadata assigned to images:")
            pprint(self.metadata, indent=2, sort_dicts=False, )

        # >  Metadata for the complete stack (not folders)
        self.scalebar_length = scalebar_length

    #
    # == Assign metadata to folder(s) ==================================

    def _check_metadata(self):
        """Checks if metadata is compatible with the object:
        - Values must have same length as folders, or be a single value
        - If single value, it's duplicated for each folder
        """

        ### shorten access
        _MD = self._metadata_perfolder

        ### Remove None values
        _MD = {k: v for k, v in _MD.items() if not v is None}

        kws = dict(target_key="folders", target_n=len(self.folder))
        for k, v in _MD.items():
            _MD[k] = ut.check_samelength_or_number(key=k, val=v, **kws)

        return _MD

    def _init_metadata(self, **metadata) -> dict:
        """Makes a dict with shortpaths as keys and metadata as values.
        each metadata is a dict with metadata keys as keys and values as
        values."""

        ### Initialize outer dict: shortpath -> metadata
        MD = {shortpath: {} for shortpath in self.path_short}

        ### Process metadata defined by arguments
        # > Check metadata and expand to full length
        metadata_raw = self._check_metadata()
        # > Fill in metadata
        for k, v in metadata_raw.items():
            for i, val in enumerate(v):
                MD[self.path_short[i]][k] = val

        # ### Fill in image names found in every folder
        for folder, names in self.imgnames.items():
            MD[folder]["imgnames"] = names

        return MD

    def _add_metadata_per_folders(self):
        """Adds metadata defined in constructor to images per folder"""

        print(type(self.imgs), self.imgs.shape)

        for i, img in enumerate(self.imgs):
            # > Get metadata for folder
            if img.folder == "unknown folder":
                continue
            metadata = self.metadata[img.folder]
            for mdkey, mdval in metadata.items():
                setattr(self.imgs[i], mdkey, mdval)

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

    Z = ImgsMeta(
        data=folders,
        fname_pattern="*DAPI*.tif",
        # invert=False,
        sort=False,
        imgname_position=[
            0,
            2,
            2,
        ],  # > Extract a key from the filename
        # pixel_length=[
        #     0.1,
        #     0.2,
        #     0.3,
        # ],
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
    print(Z.imgs[0].info)
    print(Z.imgs[3].info)
    print(Z.imgs[4].info)
