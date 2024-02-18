"""A Class extending np.array functions with simple attributes to store
image information"""

# %%
### imports
import re
from pathlib import Path
import numpy as np

# > Local
import imagep._utils.utils as ut
import imagep._configs.rc as rc
import imagep.images._array_to_str as a2s

# from imagep.images.imgs_import import ImgsImport


# %%
# ======================================================================
# == Class mdarray ========= ===========================================
class mdarray(np.ndarray):
    """A MedataData array. Extends np.array with image-properties:
    - Scale and unit [meter/pixel]
        - Calculates total width and height of the image
    - Name unique to image within a folder
    - Folder where the image is stored, preferrably a short path
      containing experimental parameters
    - etc.

    Take care:
    - Only compatible with shapes (x, y) and (z, x, y)
    """

    def __new__(
        cls,
        array: np.ndarray,
        name: str = "unnamed",
        index: tuple[int] = (0, 0),  # > (z, total images)
        folder: str | Path = "unknown folder",
        pixel_length: float = None,
        unit: str = rc.UNIT_LENGTH,
        **np_kws,
    ):
        """Using __new__ is preferred over __init__ because numpy
        arrays are immutable objects, meaning their attributes cannot be
        modified after creation. It allows you to customize the creation
        of the object before it's initialized.
        """
        obj = np.asarray(array, **np_kws).view(cls)
        obj.name = name
        obj.index = index
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
        self.name = getattr(new_obj, "name", "unnamed")
        self.index = getattr(new_obj, "index", 0)
        self.folder = getattr(new_obj, "folder", "unknown folder")
        self.pixel_length = getattr(new_obj, "pixel_length", None)
        self.unit = getattr(new_obj, "unit", rc.UNIT_LENGTH)

    @property
    def metadata(self) -> dict:
        """Returns metadata"""
        return dict(
            name=self.name,
            index=self.index,
            folder=self.folder,
            pixel_length=self.pixel_length,
            unit=self.unit,
        )

    #
    # == Information ===================================================
    def __repr__(self) -> str:
        return self._array_str(maximages=3)

    def __str__(self) -> str:
        return self._array_str()

    def _array_str(self, maximages: int = None) -> str:
        """Returns a string representation of the array."""
        if self.ndim == 0:
            return str(self.item())
        elif self.ndim == 1:
            return a2s.array1D_to_str(self)
        elif self.ndim == 2:
            return a2s.array2D_to_str(self)
        elif self.ndim == 3:
            # !! 3D array will homogenize images
            # > This is only here so less stuff breaks
            return a2s.arrays_to_str(self, maximages=maximages)
        else:
            raise ValueError(f"Array with {self.ndim} dimensions not supported")

    @property
    def info_short(self) -> str:
        just = lambda x: ut.justify_str(x, justify=6)
        info = [
            just("Name") + f"'{self.name}' from '{self.folder}'",
        ]
        return "\n".join(info)

    @property
    def info(self) -> str:
        just = ut.justify_str
        form = lambda num: ut.format_num(num, exponent=rc.FORMAT_EXPONENT)
        u = self.unit

        info = [
            str(type(self)) + " object",
            just("Name") + f"{self.name}",
            just("Folder") + f"{self.folder}",
            # form("Array type") + f"{type(self)}",
            just("Pixel type") + f"{self.dtype}",
            just("Shape [pixel]") + f"{self.shape}",
            just("Unit") + f"{self.unit}",
            just(f"Pixel-length")
            + (
                f"{form(self.pixel_length)} {u}/pixel"
                if self.pixel_length is not None
                else "No pixel length defined"
            ),
            just(f"W x H [{u}]")
            + f"{form(self.width_meter[0])} x {form(self.height_meter[0])}",
        ]
        return "\n".join(info)

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


def _test_img(img: mdarray):
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
        "",
        "img.info_short",
        img.info_short,
        "",
        "img.metadata",
        img.metadata,
        "",
        "img._array_str()",
        img._array_str(),
    ]

    for ins in instructions:
        print(ins)


if __name__ == "__main__":
    import imagep.images.importtools as importtools

    P = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/"
    ### From txt
    # path = P + "/231215_adipose_tissue/2 healthy z-stack detailed/Image4_12.txt"
    ### From
    path = P + "/240201 Imunocyto/Exp. 3/Dmp1/LTMC I D0 DAPI.tif"
    array = importtools.array_from_path(path=path)
    img = mdarray(array=array, pixel_length=2.3)
    _test_img(img)

    # %%
    ### Test __repr__
    img

    # %%
    ### Test print()
    print(img)

    # %%
    ### Test string (unreadable)
    img._array_str()

    # %%
    ### Test string representation
    arrays = [
        np.ones((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.float16) * 2.5,
        np.ones((4, 4), dtype=np.float32) * 3.5,
        np.ones((5, 5), dtype=np.uint8),
        np.ones((10, 10), dtype=np.uint8),
        np.ones((15, 15), dtype=np.uint8) * 255,
        np.ones((20, 20), dtype=np.float64),
        np.ones((30, 30), dtype=np.float16),
        np.ones((1024, 1024), dtype=np.float32),
    ]

    ### Make numbers heterogenous
    for z, img in enumerate(arrays):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                img[y, x] = img[y, x] / (x + 1) / (y + 1) * (z + 1)

    ### Add metadata
    for z, img in enumerate(arrays):
        arrays[z] = mdarray(img, pixel_length=20, name=f"testImage {z}")

    ### Print strings
    for img in arrays:
        print(img)
    # %%
    ### Convert to 3D array
    arrays_homo = [
        np.ones((5, 5), dtype=np.uint8),
        np.ones((5, 5), dtype=np.float16),
    ]
    ### Add metadata
    a3D = mdarray(arrays_homo)

    for z, img in enumerate(a3D):
        a3D[z].name = f"testImage {z}"
        a3D[z].pixel_length = 20

    # !!
    print(a3D.dtype)
    print(a3D.shape)
    print(a3D.name)
    print(a3D[0].name)
