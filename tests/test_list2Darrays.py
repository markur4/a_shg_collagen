"""Tests for the list2Darrays class"""
import numpy as np

# > Locals
import imagep._utils.types as T 
from imagep.images.list_of_arrays import list2Darrays

# %%
def _test_list2Darrays_input():
    """Tests the list2Darrays class"""
    import numpy as np

    def _test(
        arr: np.ndarray,
        target_shape,
        must_raise=False,
    ):
        try:
            loa = list2Darrays(arrays=_arr)
            print("  target_shape:\t", target_shape)
            print("  shape:\t\t", loa.shapes)
            print("  type loa\t", type(loa.arrays))
            print("  type loa[0]\t", type(loa.arrays[0]))
            print()
            assert loa.shapes == target_shape
            assert type(loa.arrays) == list
            assert isinstance(loa.arrays[0], T.array)
        except ValueError as e:
            if must_raise:
                shape = arr.shape if hasattr(arr, "shape") else arr[0].shape
                shape_str = (
                    f"shape {shape}"
                    if hasattr(arr, "shape")
                    else f"list of shape {(len(shape), *shape)}"
                )
                print(
                    f"  Correct error after passing {shape_str} to list2Darrays:"
                )
                print(" ", e)
                print()
            else:
                raise e

    print("> Single 2D array:")
    _arr = np.ones((10, 10))
    _test(_arr, (1, {10}, {10}))

    print("> List of 2D arrays:")
    _arr = [np.ones((10, 10)), np.ones((20, 20)), np.ones((30, 30))]
    _test(_arr, (3, {10, 20, 30}, {10, 20, 30}))

    print("> Single 3D array:")
    _arr = np.ones((3, 10, 10))
    _test(_arr, (3, {10}, {10}))

    print("> One 4D array, raises ValueError")
    _arr = np.ones((3, 10, 10, 3))
    _test(_arr, (3, {10}, {10}), must_raise=True)

    print("> List of 3D arrays, raises ValueError")
    _arr = [np.ones((10, 10, 10)), np.ones((20, 20, 20)), np.ones((30, 30, 30))]
    _test(_arr, (3, {10, 20, 30}, {10, 20, 30}), must_raise=True)

    print("> One 1D array, raises ValueError")
    _arr = np.ones(10)
    _test(_arr, (3, {10}, {10}), must_raise=True)


if __name__ == "__main__":
    _test_list2Darrays_input()


# %%
def _test_list2Darrays_slicing():
    loa = list2Darrays(
        arrays=[
            np.ones((20, 25)),
            np.ones((20, 25)),
            np.ones((50, 55)),
            np.ones((50, 55)),
            np.ones((50, 55)),
            np.ones((50, 55)),
        ],
    )

    def _test(
        loa_sliced: T.array | list2Darrays,
        target_shape,
        target_type,
        must_raise=False,
    ):
        shape = (
            loa_sliced.shapes
            if hasattr(loa_sliced, "shapes")
            else loa_sliced.shape
        )
        print("loa_sliced.shape:\t", shape, "\t", target_shape)
        print("type loa_sliced:\t", type(loa_sliced))
        print()
        assert isinstance(loa_sliced, target_type)
        assert shape == target_shape

    print("> Test shape without changea")
    _test(loa, (6, {20, 50}, {25, 55}), list2Darrays)

    print("> loa[int]")
    _test(loa[0], (20, 25), np.ndarray)
    _test(loa[3], (50, 55), np.ndarray)

    print("> loa[int:int]")
    _test(loa[1:3], (2, {20, 50}, {25, 55}), list2Darrays)
    _test(loa[0:6:2], (3, {20, 50}, {25, 55}), list2Darrays)

    print("> loa[int, int, int]")
    _test(loa[0, 3, 1], (3, {20, 50}, {25, 55}), list2Darrays)

    print("> loa[[int, int, int]]")
    _test(loa[[0, 3, 1]], (3, {20, 50}, {25, 55}), list2Darrays)

    print("> loa[int, int, int][int]")
    _test(loa[0, 3, 4][1], (50, 55), np.ndarray)


if __name__ == "__main__":
    _test_list2Darrays_slicing()


# %%
def _test_list2Darrays_setting():
    loa = list2Darrays(
        arrays=[
            np.ones((20, 25)),
            np.ones((20, 25)),
            np.ones((50, 55)),
            np.ones((50, 55)),
            np.ones((50, 55)),
            np.ones((50, 55)),
        ],
    )

    def _test(
        val,
        item,
        target_shape,
        target_type,
        must_raise=False,
    ):
        try:
            _loa = loa.copy()  # > reset loa
            _loa[val] = item  # > <<<<<<<<<<<<<<<

            print("  shape:\t", _loa.shapes, "\t", target_shape)
            print("  type:\t", type(_loa), target_type)
            print()
            assert isinstance(_loa, target_type)
            assert _loa.shapes == target_shape
            assert isinstance(_loa[0], T.array)
        except ValueError as e:
            shape = item.shape if hasattr(item, "shape") else item[0].shape
            shape_str = (
                f"shape {shape}"
                if hasattr(item, "shape")
                else f"list of array shape {(len(shape), *shape)}"
            )
            if must_raise:
                print(f"  Correct error after setting loa[{val}] = {shape_str}")
                print(" ", e)
                print()
            else:
                raise e
        return _loa

    print("> Test shape without change")
    _loa = _test(0, loa[0], (6, {20, 50}, {25, 55}), list2Darrays)

    print("> loa[int] = np.array 2D")
    _loa = _test(
        0, np.ones((10, 10)), (6, {10, 20, 50}, {10, 25, 55}), list2Darrays
    )

    print("> loa[int] = [np.array 2D]")
    _loa = _test(
        0, [np.ones((10, 10))], (6, {10, 20, 50}, {10, 25, 55}), list2Darrays
    )

    print("> loa[int] = [np.array 2D, np.array 2D]", "raises ValueError")
    _loa = _test(
        0,
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> loa[int] = np.array 3D", "raises ValueError")
    _loa = _test(
        0,
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> loa[int:int] = np.array 2D")
    _loa = _test(
        slice(1, 3),
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    # > Setting a range with a single array should create duplicates
    assert np.array_equal(_loa[1], _loa[2])

    print("> loa[int:int] = [np.array 2D]")
    _loa = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    assert np.array_equal(_loa[1], _loa[2])

    print("> loa[int:int] = [np.array 2D, np.array 2D]", "CORRECT")
    _loa = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((20, 20))],
        (6, {10, 20, 50}, {10, 20, 25, 55}),
        list2Darrays,
    )

    print(
        "> loa[int:int] = [np.array 2D, np.array 2D, np.array 2D]",
        "raises ValueError",
    )
    _loa = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> loa[int:int] = np.array 3D", "raises ValueError")
    _loa = _test(
        slice(1, 3),
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> loa[int, int, int] = np.array 2D", "CORRECT")
    _loa = _test(
        (0, 6, 3),
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    assert np.array_equal(_loa[0], _loa[3])

    print("> loa[int, int, int] = [np.array 2D]")
    _loa = _test(
        (0, 6, 3),
        [np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    assert np.array_equal(_loa[0], _loa[3])

    print(
        "> loa[int, int, int] = [np.array 2D, np.array 2D]", "raises ValueError"
    )
    _loa = _test(
        (0, 6, 3),
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> loa[int, int, int] = np.array 3D", "CORRECT")
    _loa = _test(
        (0, 6, 3),
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    print("> loa[[int, int, int]] = np.array 4D", "raises ValueError")
    _loa = _test(
        [0, 6, 3],
        np.ones((3, 10, 10, 3)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> loa[[int, int, int]] = np.array 2D", "CORRECT")
    _loa = _test(
        [0, 6, 3],
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )


if __name__ == "__main__":
    _loa = _test_list2Darrays_setting()


# %%
def _test_list2Darrays_methods(loa):

    # > Test loa.dtypes
    print("> loa.dtype.pop() == np.float16; ", loa.dtypes)
    assert loa.dtypes.pop() == np.float16
    print("  loa.dtype:", loa.dtypes, "\n")

    # > Test loa.shape

    # > Test loa.astype()
    print("> loa.astype()")
    loa = loa.astype(np.int32)
    assert loa.dtypes.pop() == np.int32
    print("  loa.dtype:", loa.dtypes, "\n")

    # > Test copy
    print("> loa.copy()")
    loa2 = loa.copy()
    assert loa2.shapes == loa.shapes
    assert loa2.dtypes == loa.dtypes
    print("  copied")

    # > Test asarray
    print("> loa.asarray")
    arr = loa.asarray
    assert arr.shape == loa.shape
    assert arr.dtype == np.int32


if __name__ == "__main__":

    loa_homo = list2Darrays(
        arrays=[np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))],
        dtype=np.float16,
    )
    loa_1type = list2Darrays(
        arrays=[np.ones((10, 10)), np.ones((20, 20)), np.ones((30, 30))],
        dtype=np.float16,
    )
    loa_dtypes = list2Darrays(
        arrays=[
            np.ones((10, 10), dtype=np.uint8),
            np.ones((20, 20), dtype=np.float32),
            np.ones((30, 30), dtype=np.float16),
        ],
    )

    for loa in [loa_homo, loa_1type, loa_dtypes]:
        _test_list2Darrays_methods(loa)

### Test as part of Collection
# %%
if __name__ == "__main__":
    import imagep as ip

    folder = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = ip.Collection(
        data=folder,
        fname_extension="txt",
        verbose=True,
        pixel_length=(1.5 * 115.4) / 1024,
        imgname_position=1,
    )
    I = 6

    # %%
    loar = list2Darrays(arrays=list(Z.imgs))
    loar.shapes
    # Z.imgs.tolist()
