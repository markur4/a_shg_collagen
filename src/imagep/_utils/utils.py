"""Utility functions for imagep"""
# %%
from typing import Generator

import os


import numpy as np


# %%
# == Formatting ========================================================
### Set runtime configuration for Justifying text
_JUSTIFY = 23


def justify_str(string: str, justify=20):
    return str(string + ": ").ljust(justify).rjust(justify + 2)


def _test_adjust_margin():
    print(justify_str("long string") + "just=default")
    print(justify_str("very long string") + "just=default")
    print()
    print(justify_str("module level", justify=_JUSTIFY) + "just=Module Default")
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
_EXPONENT = 2


def format_num(number: int | float, position: int =0, exponent: int = 2) -> str:
    """Formats number to scientific notation if too small or too big
    :param number: Number to format
    :type number: int | float
    :param position: Position of number in list, defaults to 0
    :type position: int
    :param exp: Threshold exponent to switch to scientific notation if
        bigger/smaller than this, defaults to 2
    :type exp: int, optional
    """
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
    print(format_num(0.01, exponent=_EXPONENT))
    print(format_num(1.2, exponent=_EXPONENT))


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


# == Performance ===================================================
def get_n_cores(utilize: int = 0.75) -> int:
    """Get the number of cores of the machine

    :param utilize: How many cores to utilize as % of total cores,
        defaults to .75
    :type utilize: int, optional
    """
    n_cores = os.cpu_count()
    n_cores = int(round(n_cores * utilize))

    return n_cores
