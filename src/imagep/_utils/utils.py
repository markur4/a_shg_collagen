"""Utility functions for imagep"""

# %%
from typing import Generator, Callable, TypedDict
import re

import os
from pathlib import Path
import time

import numpy as np

import matplotlib.pyplot as plt

# !! no Local imports
# from imagep._utils.types import mdarray
# from imagep._rc import _EXPONENT


# %%
# == Configs =========================================================
class ConfigImport(TypedDict):
    """Configuration for importing images"""

    # fname_extension: str
    sort: bool
    # sortkey: Callable
    invertorder: bool
    dtype: np.dtype
    # import_kws: dict


# TODO: make config for mpl, so the user's won't be overridden


# %%
# == Print Messages ================================================
def _print_start_message(
    msg: str,
    msg_after_points: str = "",
    cached: bool = None,
    parallel=None,
    n_cores: int = None,
) -> str:
    m = f"\t> {msg} ..."
    if msg_after_points:
        m += f" {msg_after_points}"

    if parallel:
        m += f" ({n_cores} workers)"
    if cached:
        m += " (checking cache)"

    print(m)
    return m


def _print_end_message(
    msg: str,
    msg_after_done: str = "",
    split: str = " ... ",
    dt: float | str = "",
) -> None:
    ### Remove the ">" from the message
    msg = msg.replace("> ", "  ")

    ### Remove info that comes after ": "
    # > e.g. msg = "Subtracting threshold: method=triangle, sigma=1.5"
    if split in msg:
        msg = msg.split(split)[0]
    # > delta time
    if dt:
        dt = f"({dt:.2f} s)"

    msg = f"  {msg} DONE {dt} {msg_after_done}"

    print(msg)
    print()


def _messaged_execution(
    f: Callable,
    msg: str,
    msg_after_points: str = "",
    msg_after_done: str = "",
    split: str = " ... ",
    acc_KWS: dict = dict(),
    filter_KWS: dict = dict(),
) -> np.ndarray:
    ### Start message
    msg = _print_start_message(
        msg=msg,
        msg_after_points=msg_after_points,
        **acc_KWS,
    )
    t1 = time.time()

    ### Excetute
    results = f(**filter_KWS, **acc_KWS)

    ### End message
    dt = time.time() - t1
    _print_end_message(
        msg=msg,
        msg_after_done=msg_after_done,
        dt=dt,
        split=split,
    )

    return results


# %%
# == Formatting ========================================================
def shortenpath(path: Path | str) -> str:
    """Shorten the path to the last 2 elements"""
    path = Path(path)
    return str(path.parent.name + "/" + path.name)


# %%


def justify_str(string: str, justify: int = 23, justify_r: int = None):
    justify_r = justify + 2 if justify_r is None else justify_r
    return str(string + ": ").ljust(justify).rjust(justify_r)


def _test_adjust_margin():
    print(justify_str("long string") + "just=default")
    print(justify_str("very long string") + "just=default")
    print()
    print(justify_str("module level", justify=2) + "just=Module Default")
    print()
    s = "stri"
    print()
    print(justify_str(s, justify=3) + f"just=3 (len({s})={len('test')})")
    print(justify_str(s, justify=4) + "just=4")
    print(justify_str(s, justify=5) + "just=5")
    print(justify_str(s, justify=6) + "just=6")


if __name__ == "__main__":
    _test_adjust_margin()


# %%
### Set runtime configuration for exponent
def format_num(
    number: int | float, position: int = 0, exponent: int = 2
) -> str:
    """Formats number to scientific notation if too small or too big
    :param number: Number to format
    :type number: int | float
    :param position: Position of number in list, defaults to 0
    :type position: int
    :param exp: Threshold exponent to switch to scientific notation if
        bigger/smaller than this, defaults to 2
    :type exp: int, optional
    """
    ### Return True / False if encountered
    if number == True or number == False:
        return str(number)
    
    # if isinstance(number, bool):
    #     return str(number)
    
    
    ### If number is nested in a list, get it
    if isinstance(number, (list, tuple)):
        if len(number) == 1:
            number = number[0]
        else:
            raise ValueError(f"Number is a list of length {len(number)}")


    ### Return 0 if number is 0
    if number == 0:
        return "0"
    ### Return 1 if number is exactly 1 and not a float
    if number == 1 and not isinstance(number, float):
        return "1"

    ### Get exponent
    e = np.floor(np.log10(number))

    ### Get significant number of digits after decimal point
    if isinstance(number, (int, np.uint8, np.uint16, np.uint32, np.uint64)):
        sig = 0
    elif number < 0.1:
        sig = exponent + 1
    elif 0.1 <= number < 2:
        sig = 2
    elif 2 <= number < 10:
        sig = 1
    else:
        sig = 0

    ### Format
    if e < -exponent or e > exponent:
        r = f"{number:.1e}"  # > scientific e.g. 1.5e+02

    else:
        r = f"{number:.{sig}f}"  # > fixed e.g. 150.0000
    return r


def _test_format_num():
    print(format_num(0))
    print(format_num(0.00000))
    print(format_num(1))  # > int
    print(format_num(1234))  # > int
    print(format_num(1.1))
    print(format_num(12.1))
    print(format_num(123.1))
    print(format_num(123.01))
    print(format_num(1234.1))
    print(format_num(1234.01))
    print(format_num(12345.1))
    print(format_num(0.81001))
    print(format_num(0.85001))
    print(format_num(0.50001))
    print(format_num(0.10001))
    print(format_num(0.01001))
    print(format_num(0.00101))
    print(format_num(0.00011))
    print(format_num(0.00001))
    print()
    print(format_num(0.01, exponent=2))
    print(format_num(1.2, exponent=2))


if __name__ == "__main__":
    _test_format_num()


# %%


def check_arguments(kws: dict, required: list, kws_name="kws"):
    """Check if all required keys are present in kws"""
    for k in required:
        if not k in kws.keys():
            raise KeyError(f"Missing argument '{k}' in {kws_name}: {kws}")


#
# == Image Slice =======================================================


def indices_from_slice(
    slice: str | int | list | tuple,
    n_imgs: int,
    aslist: bool = False,
) -> list[int] | range:
    """Get indices from slice"""
    if slice == "all":
        indices = range(n_imgs)
    elif isinstance(slice, int):
        indices = [slice]
    elif isinstance(slice, list):
        indices = slice
    elif isinstance(slice, tuple):
        indices = range(*slice)
    else:
        raise ValueError(
            f"img_slice must be 'all', int or tuple(start,stop,step), not {type(slice)}"
        )

    if aslist:
        indices = list(indices)

    return indices


def _test_indices_from_slice(verbose=False):
    n_imgs = 10

    def _print(r):
        print(r, type(r), list(r))

    _print(indices_from_slice("all", n_imgs))
    _print(indices_from_slice(1, n_imgs))  # > list[int]
    _print(indices_from_slice((1, 5), n_imgs))  # > range
    _print(indices_from_slice((1, 5, 2), n_imgs))  # > range
    _print(indices_from_slice((1, 5, 2), n_imgs, aslist=True))  # > list[int]


if __name__ == "__main__":
    _test_indices_from_slice()


#
# == Performance ===================================================
def toint(x: float) -> int:
    """Round to int"""
    return int(round(x))


# %%
def cores_from_percent(utilize: int = 0.75) -> int:
    """Get the number of cores of the machine

    :param utilize: How many cores to utilize as % of total cores,
        defaults to .75
    :type utilize: int, optional
    """
    n_cores = os.cpu_count()
    n_cores = toint(n_cores * utilize)

    return n_cores


def cores_from_shape(shape: tuple[int, int, int]) -> int:
    """Get the number of cpu to use from shape of image

    :param shape: Shape of array
    :type shape: tuple[int, int, int]
    """
    ### Core numbers
    cores_all = os.cpu_count()
    cores_half = toint(cores_all * 0.50)
    cores_max = cores_all - 1  # > Leave one core for other processes

    ### Get Image sizes
    PIX = np.prod(shape)
    small, big = 1024 * 1024, 1440 * 1440

    ### Get number
    cores_use = 2  # > Start with 2 cores
    ### 8 big images to 16 small: [9e6 - 16e6]
    if 8 * big <= PIX < 16 * small:
        cores_use = cores_half  # > 8 big images
    ### 16 small to 16 big: [17e6 - 34e6]
    elif 16 * small <= PIX < 16 * big:
        cores_use = cores_half  # > 16 small images
    ### 32 normal images = 16 big images: [> 34e6]
    elif 32 * small <= PIX:
        cores_use = cores_max

    ### Make sure we don't have more cores than images
    if cores_use > shape[0]:
        cores_use = min(cores_use, shape[0]) - 1

    ### Make sure at least one core is used
    cores_use = 1 if cores_use < 1 else cores_use

    return cores_use


def _test_get_n_cores_from_shape():
    f = lambda s: print(
        f"shape: {s} ",
        cores_from_shape(s),
        f"\t{s[0] / cores_from_shape(s):.2f} images per core",
    )

    print("CPU count:", os.cpu_count(), "\n")

    print("Few normal images:")
    for i in range(0, 10):
        f((i, 1024, 1024))
    print()

    print("Lots of normal images:")
    for i in range(7, 40, 3):
        f((i, 1024, 1024))
    print()

    print("Few big images:")
    for i in range(1, 10):
        f((i, 1440, 1440))

    print("Lots of big images:")
    for i in range(7, 40, 3):
        f((i, 1440, 1440))


if __name__ == "__main__":
    _test_get_n_cores_from_shape()

#
# == I/O ===============================================================


def saveplot(fname: str, verbose: bool = True) -> None:
    """Saves current plot to file"""

    if not isinstance(fname, str):
        raise ValueError(f"You must provide a filename for save. Got: {fname}")

    # > add .pdf as suffix if no suffix is present
    if "." in fname:
        fname = Path(fname)
    else:
        fname = Path(fname).with_suffix(".pdf")
    plt.savefig(fname, bbox_inches="tight")

    if verbose:
        print(f"Saved plot to: {fname.resolve()}")


#
# == Check arguments ===================================================


def check_samelength_or_number(
    key: str,
    val: int | float | list[int | float | str],
    target_key: str,
    target_n: int,
) -> list[int | float]:

    if isinstance(val, (int, float, str)):
        val = [val for _ in range(target_n)]
    elif not isinstance(val, (list, tuple)):
        raise ValueError(
            f"{key} must be a list or a single number, not {type(val)}"
        )
    elif len(val) == 1:
        val = [val[0] for _ in range(target_n)]
    elif len(val) != target_n:
        raise ValueError(
            f"'{key}' can't contain {len(val)} entries, it must be as"
            f" long as '{target_key}' (has {target_n} entries)"
        )
    return val
