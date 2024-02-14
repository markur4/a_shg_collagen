"""Tests for the list2Darrays class"""

# %%
from typing import Callable
import numpy as np

from pprint import pprint

# > Locals
import imagep._utils.types as T
from imagep.images.list2Darrays import list2Darrays


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
            larry = list2Darrays(arrays=_arr)
            print("  target_shape:\t", target_shape)
            print("  shape:\t\t", larry.shapes)
            print("  type larry\t", type(larry.arrays))
            print("  type larry[0]\t", type(larry.arrays[0]))
            print()
            assert larry.shapes == target_shape
            assert type(larry.arrays) == list
            assert isinstance(larry.arrays[0], T.array)
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

    print("> List of single 3D array:")
    _arr = [np.ones((3, 10, 10))]
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
    larry = list2Darrays(
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
    _test(larry, (6, {20, 50}, {25, 55}), list2Darrays)

    print("> larry[int]")
    _test(larry[0], (20, 25), np.ndarray)
    _test(larry[3], (50, 55), np.ndarray)

    print("> larry[int:int]")
    _test(larry[1:3], (2, {20, 50}, {25, 55}), list2Darrays)
    _test(larry[0:6:2], (3, {20, 50}, {25, 55}), list2Darrays)

    print("> larry[int, int, int]")
    _test(larry[0, 3, 1], (3, {20, 50}, {25, 55}), list2Darrays)

    print("> larry[[int, int, int]]")
    _test(larry[[0, 3, 1]], (3, {20, 50}, {25, 55}), list2Darrays)

    print("> larry[int, int, int][int]")
    _test(larry[0, 3, 4][1], (50, 55), np.ndarray)


if __name__ == "__main__":
    _test_list2Darrays_slicing()


# %%
def _test_list2Darrays_setting():
    larry = list2Darrays(
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
            _larry = larry.copy()  # > reset larry
            _larry[val] = item  # > <<<<<<<<<<<<<<<

            print("  shape:\t", _larry.shapes, "\t", target_shape)
            print("  type:\t", type(_larry), target_type)
            print()
            assert isinstance(_larry, target_type)
            assert _larry.shapes == target_shape
            assert isinstance(_larry[0], T.array)
        except ValueError as e:
            shape = item.shape if hasattr(item, "shape") else item[0].shape
            shape_str = (
                f"shape {shape}"
                if hasattr(item, "shape")
                else f"list of array shape {(len(shape), *shape)}"
            )
            if must_raise:
                print(
                    f"  Correct error after setting larry[{val}] = {shape_str}"
                )
                print(" ", e)
                print()
            else:
                raise e
        return _larry

    print("> Test shape without change")
    _larry = _test(0, larry[0], (6, {20, 50}, {25, 55}), list2Darrays)

    print("> larry[int] = np.array 2D")
    _larry = _test(
        0, np.ones((10, 10)), (6, {10, 20, 50}, {10, 25, 55}), list2Darrays
    )

    print("> larry[int] = [np.array 2D]")
    _larry = _test(
        0, [np.ones((10, 10))], (6, {10, 20, 50}, {10, 25, 55}), list2Darrays
    )

    print("> larry[int] = [np.array 2D, np.array 2D]", "raises ValueError")
    _larry = _test(
        0,
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> larry[int] = np.array 3D", "raises ValueError")
    _larry = _test(
        0,
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> larry[int:int] = np.array 2D")
    _larry = _test(
        slice(1, 3),
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    # > Setting a range with a single array should create duplicates
    assert np.array_equal(_larry[1], _larry[2])

    print("> larry[int:int] = [np.array 2D]")
    _larry = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    assert np.array_equal(_larry[1], _larry[2])

    print("> larry[int:int] = [np.array 2D, np.array 2D]", "CORRECT")
    _larry = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((20, 20))],
        (6, {10, 20, 50}, {10, 20, 25, 55}),
        list2Darrays,
    )

    print(
        "> larry[int:int] = [np.array 2D, np.array 2D, np.array 2D]",
        "raises ValueError",
    )
    _larry = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> larry[int:int] = np.array 3D", "raises ValueError")
    _larry = _test(
        slice(1, 3),
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> larry[int, int, int] = np.array 2D", "CORRECT")
    _larry = _test(
        (0, 6, 3),
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    assert np.array_equal(_larry[0], _larry[3])

    print("> larry[int, int, int] = [np.array 2D]")
    _larry = _test(
        (0, 6, 3),
        [np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    assert np.array_equal(_larry[0], _larry[3])

    print(
        "> larry[int, int, int] = [np.array 2D, np.array 2D]",
        "raises ValueError",
    )
    _larry = _test(
        (0, 6, 3),
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> larry[int, int, int] = np.array 3D", "CORRECT")
    _larry = _test(
        (0, 6, 3),
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )
    print("> larry[[int, int, int]] = np.array 4D", "raises ValueError")
    _larry = _test(
        [0, 6, 3],
        np.ones((3, 10, 10, 3)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
        must_raise=True,
    )

    print("> larry[[int, int, int]] = np.array 2D", "CORRECT")
    _larry = _test(
        [0, 6, 3],
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        list2Darrays,
    )


if __name__ == "__main__":
    _larry = _test_list2Darrays_setting()


# %%
def _test_list2Darrays_methods(
    larry: list2Darrays, homo_shape=True, homo_type=True
):

    # > Test larry.dtypes
    print("> larry.dtype.pop() == np.float16; ", larry.dtypes)
    assert larry.dtype == np.float64
    print("  larry.dtype:", larry.dtypes, "\n")

    # > Test larry.shape

    # > Test larry.astype()
    print("> larry.astype()")
    _larry = larry.astype(np.int32)
    assert _larry.dtype == np.int32
    assert larry.dtype == np.float64
    print("  larry.dtype:", larry.dtypes, "\n")

    # > Test copy
    print("> larry.copy()")
    loa2 = larry.copy()
    assert loa2.shapes == larry.shapes
    assert loa2.dtypes == larry.dtypes
    print("  copied", "\n")

    # > Test asarray
    print("> larry.asarray(dtype=np.int32)")
    try:
        arr = larry.asarray(dtype=np.int32)
        assert arr.shape == larry.shape
        assert arr.dtype == np.int32
        assert larry.dtype == np.float64
    except ValueError as e:
        if not homo_shape:
            print("  Correct Error when trying to make an inhomogenous array")
            print(" ", e, "\n")

    # > Test min max
    print("> larry.min, larry.max")
    larry.min(), larry.max()
    print()

    print("\n DONE \n\n")


if __name__ == "__main__":

    loa_homo = list2Darrays(
        arrays=[np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))],
        dtype=np.float64,
    )
    loa_hetero_type = list2Darrays(
        arrays=[np.ones((10, 10)), np.ones((20, 20)), np.ones((30, 30))],
        dtype=np.float64,
    )
    loa_hetero_total = list2Darrays(
        arrays=[
            np.ones((10, 10), dtype=np.uint8),
            np.ones((20, 20), dtype=np.float64),
            np.ones((30, 30), dtype=np.float16),
        ],
    )
    larries = [loa_homo, loa_hetero_type, loa_hetero_total]
    homo_shapes = [True, False, False]
    homo_dtypes = [True, True, False]

    args = zip(
        larries,
        homo_shapes,
        homo_dtypes,
    )

    ### EXECUTE
    for larry, h_shape, h_dtype in args:
        _test_list2Darrays_methods(larry, h_shape, h_dtype)
        # %%
        ### Check warnings on shapes manually
        print(loa_hetero_total.shapes)
        print(loa_hetero_total.shapes[2])
        print(loa_hetero_total.shape)
        # %%
        ### Check warnings on dtypes manually
        print(loa_hetero_total.dtypes)
        print(loa_hetero_total.dtype)


# %%
def _assert_array_similarity(
    arr1: np.ndarray | list2Darrays, arr2: np.ndarray
) -> bool:
    """Checks similarity of two similar shaped 3D arrays value by value"""
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Arrays must be similar shape: {arr1.shape}, {arr2.shape}"
        )

    for z, (z1, z2) in enumerate(zip(arr1, arr2)):
        for y, (y1, y2) in enumerate(zip(z1, z2)):
            for x, (x1, x2) in enumerate(zip(y1, y2)):
                x1 = round(x1, 5)
                x2 = round(x2, 5)
                if x1 != x2:
                    raise AssertionError(
                        f"Arrays differ at position ({z},{y},{x}):"
                        f" {x1}, {x2}"
                    )
    return True


def _test_list2Darrays_mathoperations(
    larry: list2Darrays,
    other: list | int | float | T.array | list2Darrays,
    result: str | list2Darrays = "exception",
):
    def _assertions():
        print("  ", "> ASSERTING ")
        ### Assert type
        print("   Type", type(_result_larry), type(larry))
        assert type(_result_larry) == type(larry)
        ### Assert no conversion
        assert isinstance(_result_larry, list2Darrays)

        ### Get result
        _result_array = _get_result_asarray(
            _operation,
            larry,
            other,
        )
        ### Assert Shape
        print("  Shapes:", _result_larry.shape, _result_array.shape)
        assert _result_larry.shape == _result_array.shape

        ### Check similarity value by value
        print("  Value by value")
        print("=== larry result: ===\n", _result_larry)
        print("=== array result: ===\n", _result_array)
        _assert_array_similarity(_result_larry, _result_array)

        print("> TEST PASSED \n")

    try:
        print("> op:larry + other  (Addition)")
        _operation = lambda x1, x2: x1 + x2
        _result_larry = larry + other
        _assertions()

        # print("x1 - x2")
        # _larry = larry - other
        # _assertions()

        # print("")
        # _assertions()

    except ValueError as e:
        # if result == "exception":
        #     print("  Correct error after operation")
        # else:
        #     raise e
        raise e


def _test_list2Darrays_mathoperations_multiple(
    larries: list[list2Darrays],
    others: list[int | float | T.array | list2Darrays],
    # results: list[list2Darrays],
):
    for larry in larries:
        print("=> ======= TEST SET: =========")
        print("=> larry:", larry)
        print()
        # for other, result in zip(others, results):
        for other in others:
            print("> NEW TEST:")
            print("> other:", other)
            # _test_list2Darrays_mathoperations(larry, other, result)
            _test_list2Darrays_mathoperations(larry, other)
        print()
        print()


def _get_result_asarray(op, larry: list2Darrays, other: int | float | T.array):
    """Converts larry to array and performs operation. If larry is
    inhomogenous, it iterates through elements and performs operations
    elementwise"""

    if larry.homogenous:
        return op(larry.asarray(), other)
    else:
        for i, arr in enumerate(larry):
            if isinstance(other, (int, float)):
                larry[i] = op(arr, other)
            elif isinstance(other, T.array):
                if larry[i].shape == other[i].shape:
                    larry[i] = op(arr, other[i])
    return larry

if __name__ == "__main__":

    ### Define larries
    loa_homo_s = list2Darrays(
        arrays=[np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))],
        dtype=np.float64,
    )
    loa_homo_s_heterotype = list2Darrays(
        arrays=[
            np.ones((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float16),
        ]
    )
    loa_hetero_s_type = list2Darrays(
        arrays=[np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4))],
        dtype=np.float64,
    )
    loa_hetero_s_total = list2Darrays(
        arrays=[
            np.ones((2, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.float64),
            np.ones((4, 4), dtype=np.float16),
        ],
    )

    ### Collect larries
    larries_s = [
        loa_homo_s,
        loa_homo_s_heterotype,
    ]
    # homo_s_shapes = [True, False, False]
    # homo_s_dtypes = [True, True, False]

    larries_s_all = [
        *larries_s,
        loa_hetero_s_type,
        loa_hetero_s_total,
    ]

    ### Others
    others_scalars = [
        0,
        1,
        2,
        0.5,
        np.float16(0.5),
    ]

    # %%
    ### TEST: Scalars with all larries
    _test_list2Darrays_mathoperations_multiple(
        larries=larries_s_all,
        others=others_scalars,
    )

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
    print(loar.shapes)
    print(loar.shapes[2])
    print(loar.shape)

    # %%
    # Z.imgs.tolist()
