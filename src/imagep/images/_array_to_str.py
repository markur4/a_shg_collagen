"""Make printable string representations of arrays."""

# %%
import re
import numpy as np


# > Local
# from imagep.images.mdarray import mdarray
# import imagep._utils.types as T
import imagep._utils.utils as ut

# %%
### Define tab of every Row
T = " " * 4

### Format numbers
form = lambda x: ut.format_num(x, exponent=2)


# %%


def shorten_type(typ: type) -> str:
    # > make a short version of the type
    return str(typ).split(".")[-1].replace("'>", "")


def shorten_digits(row: str) -> str:
    ### Remove apostrophes '
    row = str(row).replace("'", "")

    ### Replace "0." with "." from floats
    row = row.replace("0.", ".")
    row = row.replace("-0", "-")
    row = row.replace(",", "")

    ### Justify digits
    # > Default: e.g. 0-255 (uint8), default
    rj = 3
    # > Increase for floats
    if "." in row:
        rj = 4
    if "False" in row or "True" in row:
        rj = 5
    # > Increase for small numbers with exponents
    if re.search(r"\d+e-\d", row):
        rj = 6
    # > execute
    row = row.split(" ")
    row = [x.ljust(rj) for x in row]
    row = " ".join(row)
    return row


def get_metadata_list(array: np.ndarray) -> list[str]:
    info = array.info_short.split("\n")
    tab = " " * (len(T) // 2)
    return [f"{tab}{x}" for x in info]


def makerow(row) -> str:
    # > format row
    row = [form(x) for x in row]
    maxwidth = 6
    maxwidth_h = int(round(maxwidth / 2))
    if len(row) > maxwidth:
        row1 = str(row[:maxwidth_h])
        row2 = str(row[-maxwidth_h:])
        row1 = row1.replace("]", "").replace("[", "")
        row2 = row2.replace("]", "").replace("[", "")
        row1 = shorten_digits(row1)
        row2 = shorten_digits(row2)

        # > Assemble
        row = f"[ {row1} ... {row2} ]"
    else:
        row = str(row).replace("]", "").replace("[", "")
        row = shorten_digits(row)
        # > Assemble
        row = f"[ {row} ]"

        ### If floats, adjust every number to the same length
    return row


def finalize_row(array: str, y: int, lj: int) -> str:
    row = f"{T} {makerow(array[y])}".ljust(lj) + f" y={y}"
    return row


# %%
def array1D_to_str(array: np.ndarray):
    """Returns a string representation of the array"""
    ### Make head row
    typ = shorten_type(type(array))
    head = f" 1D {typ} length {array.shape[0]} (z) {array.dtype}:"
    ### Make first and last row
    row1 = f"{T}{makerow(array)}"
    ### Add metadata
    if hasattr(array, "info_short"):
        md = get_metadata_list(array)
        ### Insert after head and before array
        S = "\n".join([head] + md + [row1])
    else:
        S = "\n".join([head, row1])
    return S


if __name__ == "__main__":
    arrays = [
        np.ones(2, dtype=np.uint8),
        np.ones(2, dtype=np.float16) * 2.5,
        np.ones(4, dtype=np.float32) * 3.5,
        np.ones(5, dtype=np.uint8),
        np.ones(10, dtype=np.uint8),
        np.ones(15, dtype=np.uint8) * 255,
        np.ones(20, dtype=np.float64),
        np.ones(30, dtype=np.float16),
        np.ones(1024, dtype=np.float32),
    ]
    # ### Make numbers heterogenous
    for z, img in enumerate(arrays):
        for y in range(img.shape[0]):
            img[y] = img[y] / (y + 1) * (z + 1)

    for a in arrays:
        # print(a)
        print(array1D_to_str(a))


# %%
def array2D_to_str(array: np.ndarray):
    """Returns a string representation of the array"""

    ### MAIN
    def makerows_small():
        """Returns a list of rows for small images"""
        rows = [finalize_row(array, y, lj) for y in range(1, TOTALROWS - 1)]
        return [head, row1] + rows + [row4]

    def makerows_big():
        """Returns a list of rows for BIG images"""
        # > Thresholds for how many first and last rows to display
        maxrows1 = int(round(MAXROWS / 2))
        maxrows2 = TOTALROWS - maxrows1
        # > First intermediate rows
        rows2 = [finalize_row(array, y, lj) for y in range(1, maxrows1)]
        # > Last intermediate rows
        rows3 = [
            finalize_row(array, y, lj) for y in range(maxrows2, TOTALROWS - 1)
        ]
        # > Dots
        dot_start1 = len(rows2[0].split("...")[0])
        dot_start2 = len(rows3[0].split("...")[0])
        dot_extend_length = abs(dot_start1 - dot_start2)
        dots = "..." + "." * dot_extend_length
        dots = dots.rjust(dot_start1 + dot_extend_length + 3)

        return [head, row1] + rows2 + [dots] + rows3 + [row4]

    TOTALROWS = array.shape[0]

    ### Make head row
    typ = shorten_type(type(array))
    head = f" {typ} {array.shape[0]}x{array.shape[1]} (y,x) {array.dtype}:"

    ### Make first and last row
    row1 = f"{T}[{makerow(array[0])}"
    row4 = f"{T} {makerow(array[-1])}]"
    # > Add y=0 and y=last
    lj = max(len(row1), len(row4), len(head))
    # head = head.ljust(lj) + f" [y]"
    row1 = row1.ljust(lj) + f" y=0"
    row4 = row4.ljust(lj) + f" y={TOTALROWS-1}"

    ### Make inbetween rows
    MAXROWS = 6  # !! Even Numbers!
    if TOTALROWS < MAXROWS:
        rows = makerows_small()
    else:
        rows = makerows_big()

    ### Add metadata
    if hasattr(array, "info_short"):
        md = get_metadata_list(array)
        ### Insert after head and before array
        rows = rows[:1] + md + rows[1:]

    S = "\n" + "\n".join(rows)
    return S


if __name__ == "__main__":
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
    # ### Make numbers heterogenous
    for z, img in enumerate(arrays):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                img[y, x] = img[y, x] / (x + 1) / (y + 1) * (z + 1)

    for a in arrays:
        # print(a)
        print(array2D_to_str(a))


# %%
def array3D_to_str(arrays: np.ndarray, maximages: int = None) -> str:
    """Returns a string representation of the object"""

    ### Decide which images to show
    maximages = len(arrays) if maximages is None else maximages
    indices = list(range(len(arrays)))
    # > If there are too many images, show first, middle and last
    toomany = len(arrays) > maximages
    if toomany:
        indices = [0, len(arrays) // 2, len(arrays) - 1]

    ### OBJECT HEADER
    S = f"{len(arrays)} images:"
    ### ARRAY CONTENT
    for i in indices:
        ### get image
        img = arrays[i]

        ### Get rows per 2Darray
        rows = array2D_to_str(img).split("\n")

        ### Make Head row
        rows[1] = f" #{i+1}/{len(arrays)}:" + rows[1]

        ### Add ... between first, middle and last
        if toomany:
            rows += [f"..."]

        ### Add to string
        S += "\n".join(rows)

    return S


if __name__ == "__main__":
    print(array3D_to_str(arrays))
    # %%
    ### Test maximages
    print(array3D_to_str(arrays, maximages=3))
