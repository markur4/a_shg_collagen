"""A Class extending np.array functions with simple attributes to store
image information"""

# %%
### imports
from pathlib import Path

import numpy as np

# > Local
import imagep._utils.utils as ut
import imagep._rc as rc
# from imagep.images.imgs_import import ImgsImport



# %%
# ======================================================================
# == Class ArrayWithMetadata ===========================================
class Mdarray(np.ndarray):
    """A MedataData array. Extends np.array with image-properties:
    - Scale and unit [meter/pixel]
        - Calculates total width and height of the image
    - Name unique to image within a folder
    - Folder where the image is stored, preferrably a short path
      containing experimental parameters
    - etc.
    """

    def __new__(
        cls,
        array: np.ndarray,
        name: str = "unnamed",
        folder: str | Path = "unknown folder",
        pixel_length: float = None,
        unit: str = rc.UNIT_LENGTH,
    ):
        """Using __new__ is preferred over __init__ because numpy
        arrays are immutable objects, meaning their attributes cannot be
        modified after creation. It allows you to customize the creation
        of the object before it's initialized.
        """
        obj = np.asarray(array).view(cls)
        obj.name = name
        obj.folder = folder
        obj.pixel_length = pixel_length
        obj.unit = unit
        return obj

    def __array_finalize__(self, new_obj) -> None:
        """Modifies instances of np.ndarray that's used to initialize
        Img.
        - When a new array object is created from an existing array,
          numpy checks if the class of the new array (Subclass Img)
          defines a __array_finalize__ method.
        - numpy calls this method with the new array object as an
          argument.
        - !!! Inside the __array_finalize__ method, `self` references to the
          original array object !!!
        - This modifies the original object according to the subclass.
        """

        if new_obj is None:
            return None

        ### self references to the original np.array!
        self.name = getattr(new_obj, "name", "")
        self.folder = getattr(new_obj, "folder", "")
        self.pixel_length = getattr(new_obj, "pixel_length", None)
        self.unit = getattr(new_obj, "unit", rc.UNIT_LENGTH)
    
    #
    # == Information ===================================================
    def __repr__(self) -> str:
        """Since I don't like that np.ndarray prints out a huge array
        every time I call it, override that with a more concise version.
        Retrive only first and last single rows of the array, the dtype,
        the name and the folder.
        """
        return self.info
            
        
        
    @property
    def info(self) -> str:
        form = ut.justify_str
        u = self.unit

        info = [
            type(self).__name__ + " object",
            form("Name") + f"{self.name}",
            form("Folder") + f"{self.folder}",
            form("Array type") + f"{type(self)}",
            form("Pixel type") + f"{self.dtype}",
            form("Shape") + f"{self.shape} pixels",
            form("Unit") + f"{self.unit}",
            form("Pixel length")
            + (
                f"{self.pixel_length} {u}/pixel"
                if self.pixel_length is not None
                else "No pixel length defined"
            ),
            form("Width, Height") + f"{self.width_meter[0]} x {self.height_meter[0]} {u}",
        ]
        return "\n".join(info)

    @property
    def metadata(self) -> dict:
        """Returns metadata"""
        return dict(
            name=self.name,
            folder=self.folder,
            pixel_length=self.pixel_length,
            unit=self.unit,
        )

    #
    # == Properties ====================================================

    @property
    def width_meter(self) -> tuple[float, str] | str:
        """Returns a tuple with the total width of the image and the
        length unit. If no pixel length is defined, it returns a message."""
        if self.pixel_length is None:
            return "No pixel length defined"
        else:
            width = self.shape[1] * self.pixel_length
            return width, self.unit

    @property
    def height_meter(self) -> tuple[float, str] | str:
        """Returns a tuple with the total width of the image and the
        length unit. If no pixel length is defined, it returns a message."""
        if self.pixel_length is None:
            return "No pixel length defined"
        else:
            height = self.shape[0] * self.pixel_length
            return height, self.unit

    # !! == End Class ==================================================


def _test_img(img: Mdarray):
    instructions = [
        ">>> type(img)",
        type(img),
        "",
        ">>> Retrieve np.array attributes (not in Img)",
        "img.shape:",
        img.shape,
        "img.dtype:",
        img.dtype,
        "",
        ">>> img.width_meter",
        img.width_meter,
        "",
        ">>> Call __repr__()",
        img,
        "",
        ">>> img + img",
        img + img,
        "",
        "img.info",
        img.info,
    ]

    for ins in instructions:
        print(ins)


if __name__ == "__main__":
    import imagep.images.importtools as importtools
    P = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/"
    ### From txt
    # path = P + "/231215_adipose_tissue/2 healthy z-stack detailed/Image4_12.txt"
    ### From
    path = P + "/240201 Imunocyto/Exp. 3 (im Paper)/Dmp1/LTMC I D0 DAPI.tif"
    array = importtools.array_from_path(path=path)
    img = Mdarray(array=array, pixel_length=2.3)
    _test_img(img)

    # %%
    ### Test if array is returned when referencing the image
    img
