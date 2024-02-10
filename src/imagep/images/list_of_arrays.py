"""A Class to store arrays of different sizes (and types) by replacing
the outermost dimension of the image stacks with a list."""
#%%
import numpy as np

#
# ======================================================================
# == Class ListOfArrays ================================================
class ListOfArrays:
    """An Alternative to np.ndarrays / ip.Mdarray compatible with
    storing images of different sizes and types. Provides a
    np.array-esque experience, giving access to properties like shape and
    dtype without complicating subsequent code.
    """

    def __init__(self, larry: list[np.ndarray]):
        if not isinstance(larry, list) and isinstance(larry[0], np.ndarray):
            raise TypeError(
                f"Must pass list of numpy arrays, not '{type(larry)}'"
            )

        ### larry = list of Arrays
        self.larry = larry

    @property
    def shape(self):
        """Returns (z, {y}, {x}), with x and y sets of all widths and
        heights occurring in the data"""
        y = {img.shape[0] for img in self.larry}
        x = {img.shape[1] for img in self.larry}
        return (len(self.larry), y, x)

    def __getitem__(self, value: slice):
        return self.larry[value]

    def __len__(self):
        return len(self.larry)

    @property
    def dtype(self):
        return self.larry[0].dtype if self.larry else None

    def astype(self, dtype: np.dtype):
        return [img.astype(dtype) for img in self.larry]

    # !! == End  Class =================================================


if __name__ == "__main__":
    import imagep as ip

    folder = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = ip.Imgs(
        data=folder, fname_extension="txt", verbose=True, x_µm=1.5 * 115.4
    )
    I = 6

    # %%
    loar = ListOfArrays(larry=list(Z.imgs))
    loar.shape
    # Z.imgs.tolist()
