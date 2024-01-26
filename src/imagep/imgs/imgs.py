"""A Class for basic funtionalities of images.
- Import
- Pixel scale to µm
- Add Scalebar
- Plotting mip
- 
"""
# %%
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

import imagep.imgs.importutils as importutils

# %%
# == Class Imgs ========================================================


class Imgs:
    def __init__(
        self,
        path: str | Path,
        x_µm: float = 200.0,
        verbose: bool = True,
    ) -> None:
        """Basic Image-stack funcitonalities.
        :param path: pathlike string or object
        :type path: str | Path
        :param x_µm: Total width of image in µm, defaults to 200
        :type x_µm: float, optional
        """
        self.verbose = verbose

        ### Folder of the image stack
        self.path = Path(path)
        self._check_path()

        ### Import Images
        if self.verbose:
            print(f"=> Importing Images from {self.path_short}...")
        self.imgs: np.ndarray = self.import_imgs(self.path)
        if self.verbose:
            print("   Importing Images Done")

        ### Totalwidth, height, depth in µm
        self.x_µm = x_µm
        self.y_µm = self.imgs.shape[1] * self.x_µm / self.imgs.shape[2]
        self.pixel_size = self.x_µm / self.imgs.shape[2]
        self.spacing = (self.x_µm, self.y_µm)

    #
    # == Import ========================================================

    def _check_path(self) -> None:
        """Check if path is valid"""
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    @staticmethod
    def import_imgs(path: Path) -> np.ndarray:
        """Import z-stack from a folder"""

        ### Get all txt files
        txts = list(path.glob("*.txt"))

        ### sort txts by number
        txts = sorted(txts, key=lambda x: int(x.stem.split("_")[-1]))

        ### Invert, since the first image is the bottom one
        txts = txts[::-1]

        ### Import all txt files
        imgs = []
        for txt in txts:
            imgs.append(importutils.from_txt(txt))

        ### Convert to numpy array
        imgs = np.array(imgs)

        return imgs


    #
    # == Retrieve Images ===============================================
    
    @property
    def stack_raw(self) -> np.ndarray:
        return self.import_imgs()

    def __iter__(self):
        return iter(self.imgs)

    def __getitem__(self, val: slice) -> np.ndarray:
        return self.imgs[val]

    