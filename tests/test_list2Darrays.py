"""Tests for the list2Darrays class"""

# %%
from typing import Callable
import numpy as np
import math

from pprint import pprint

# > Locals
import imagep.types as T
from imagep.arrays.mdarray import mdarray
from imagep.arrays.l2Darrays import l2Darrays
import imagep._example_data.larries as ed_larries


# %%
# ======================================================================
# == Definition ========================================================
def _test_list2Darrays_input():
    """Tests the list2Darrays class"""
    import numpy as np

    def _test(
        arr: np.ndarray,
        target_shape,
        must_raise=False,
    ):
        try:
            larry = l2Darrays(arrays=_arr)
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

    print("> Single 1D array:")
    _arr = np.ones(10)
    # print(_arr)
    # print([_arr])
    # _test(_arr, (1, {1}, {10}))
    _test(_arr, (1, {10}))

    print("> List of 1D array:")
    _arr = [np.ones(10)]
    # _test(_arr, (1, {1}, {10}))
    _test(_arr, (1, {10}))

    print("> List of 1D arrays:")
    _arr = [np.ones(10), np.ones(20), np.ones(30)]
    # _test(_arr, (3, {1}, {10, 20, 30}))
    _test(_arr, (3, {10, 20, 30}))

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


if __name__ == "__main__":
    _test_list2Darrays_input()


# %%
# ======================================================================
# == __get_item__ ======================================================
def _test_list2Darrays_slicing():
    larry = l2Darrays(
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
        loa_sliced: T.array | l2Darrays,
        target_shape,
        target_type,
        # must_raise=False,
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
    _test(larry, (6, {20, 50}, {25, 55}), l2Darrays)

    print("> larry[int]")
    _test(larry[0], (20, 25), np.ndarray)
    _test(larry[3], (50, 55), np.ndarray)

    print("> larry[int:int]")
    _test(larry[1:3], (2, {20, 50}, {25, 55}), l2Darrays)
    _test(larry[0:6:2], (3, {20, 50}, {25, 55}), l2Darrays)

    print("> larry[int, int, int]")
    _test(larry[0, 3, 1], (3, {20, 50}, {25, 55}), l2Darrays)

    print("> larry[[int, int, int]]")
    _test(larry[[0, 3, 1]], (3, {20, 50}, {25, 55}), l2Darrays)

    print("> larry[int, int, int][int]")
    _test(larry[0, 3, 4][1], (50, 55), np.ndarray)


if __name__ == "__main__":
    _test_list2Darrays_slicing()


# %%
# ======================================================================
# == __set_item__ ======================================================
def _test_list2Darrays_setting():
    larry = l2Darrays(
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
            if isinstance(item, list):
                shape = item[0].shape
            elif hasattr(item, "shape"):
                shape = item.shape
            elif isinstance(item, (int, float)) or np.isscalar(item):
                shape = "scalar"
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
    _larry = _test(0, larry[0], (6, {20, 50}, {25, 55}), l2Darrays)

    print("> larry[int] = np.array 2D")
    _larry = _test(
        0, np.ones((10, 10)), (6, {10, 20, 50}, {10, 25, 55}), l2Darrays
    )

    print("> larry[int] = [np.array 2D]")
    _larry = _test(
        0, [np.ones((10, 10))], (6, {10, 20, 50}, {10, 25, 55}), l2Darrays
    )

    print("> larry[int] = [np.array 2D, np.array 2D]", "raises ValueError")
    _larry = _test(
        0,
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
        must_raise=True,
    )

    print("> larry[int] = np.array 3D", "raises ValueError")
    _larry = _test(
        0,
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {25, 55}),
        l2Darrays,
        must_raise=True,
    )

    print("> larry[int:int] = np.array 2D")
    _larry = _test(
        slice(1, 3),
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
    )
    # > Setting a range with a single array should create duplicates
    assert np.array_equal(_larry[1], _larry[2])

    print("> larry[int:int] = [np.array 2D]")
    _larry = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
    )
    assert np.array_equal(_larry[1], _larry[2])

    print("> larry[int:int] = [np.array 2D, np.array 2D]", "CORRECT")
    _larry = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((20, 20))],
        (6, {10, 20, 50}, {10, 20, 25, 55}),
        l2Darrays,
    )

    print(
        "> larry[int:int] = [np.array 2D, np.array 2D, np.array 2D]",
        "raises ValueError",
    )
    _larry = _test(
        slice(1, 3),
        [np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
        must_raise=True,
    )

    print("> larry[int:int] = np.array 3D", "raises ValueError")
    _larry = _test(
        slice(1, 3),
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {25, 55}),
        l2Darrays,
        must_raise=True,
    )

    ### replace with integer
    print("> larry[int:int:int] = 1")
    _larry = _test(slice(0, 5, 2), 1, (6, {20, 50}, {25, 55}), l2Darrays)

    print("> larry[int, int, int] = np.array 2D", "CORRECT")
    _larry = _test(
        (0, 6, 3),
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
    )
    assert np.array_equal(_larry[0], _larry[3])

    print("> larry[int, int, int] = [np.array 2D]")
    _larry = _test(
        (0, 6, 3),
        [np.ones((10, 10))],
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
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
        l2Darrays,
        must_raise=True,
    )

    print("> larry[int, int, int] = np.array 3D", "CORRECT")
    _larry = _test(
        (0, 6, 3),
        np.ones((3, 10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
    )
    print("> larry[[int, int, int]] = np.array 4D", "raises ValueError")
    _larry = _test(
        [0, 6, 3],
        np.ones((3, 10, 10, 3)),
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
        must_raise=True,
    )

    print("> larry[[int, int, int]] = np.array 2D", "CORRECT")
    _larry = _test(
        [0, 6, 3],
        np.ones((10, 10)),
        (6, {10, 20, 50}, {10, 25, 55}),
        l2Darrays,
    )


if __name__ == "__main__":
    _larry = _test_list2Darrays_setting()


# %%
# ======================================================================
# == Methods ===========================================================
def _test_list2Darrays_methods(
    larry: l2Darrays, homo_shape=True, homo_type=True
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
    minim, maxim = larry.min(), larry.max()
    print(larry)
    print(type(minim), type(maxim))
    print(minim, maxim)
    assert np.isscalar(minim) and np.isscalar(maxim)

    print("\n DONE \n\n")


if __name__ == "__main__":

    # print(list(ed_larries.ARGSS))
    for larry, h_shape, h_dtype in ed_larries.ARGSS:
        _test_list2Darrays_methods(larry, h_shape, h_dtype)
    # %%
    ### Check warnings on shapes manually
    _larry_het = ed_larries.larry_hetero_total
    print(_larry_het.shapes)
    print(_larry_het.shapes[2])
    print(_larry_het.shape)
    # %%
    ### Check warnings on dtypes manually
    print(_larry_het.dtypes)
    print(_larry_het.dtype)

    # %%
    ### test min max methods
    print(_larry_het)
    _larry_het.max()
    # %%
    [img for img in _larry_het]

    # %%


# %%
# ======================================================================
# == OPERATIONS ========================================================
def _assert_value_by_value(
    arr1: np.ndarray | l2Darrays, arr2: np.ndarray
) -> bool:
    """Checks similarity of two similar shaped 3D arrays value by value"""
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Arrays must be similar shape: {arr1.shape}, {arr2.shape}"
        )

    for z, (z1, z2) in enumerate(zip(arr1, arr2)):
        for y, (y1, y2) in enumerate(zip(z1, z2)):
            for x, (x1, x2) in enumerate(zip(y1, y2)):
                if np.isnan(x1) and np.isnan(x2):
                    continue
                elif not math.isclose(x1, x2, rel_tol=1e-3, abs_tol=1e-3):
                    raise AssertionError(
                        f"Arrays differ at position ({z},{y},{x}):"
                        f" {x1}, {x2}"
                    )
    return True


def _test_list2Darrays_math_operations(
    larry: l2Darrays,
    other: list | int | float | T.array | l2Darrays,
):
    def _assertions():
        print("ASSERTIONS: ")

        ### Assert type
        print(
            "ASSERTING Type didn't change: ", type(_result_larry), type(larry)
        )
        assert type(_result_larry) == type(larry)

        print(
            "ASSERTING Array type changed? ",
            type(_result_larry[0]),
            type(larry[0]),
        )
        assert type(_result_larry[0]) == type(larry[0])

        if not _result_larry.dtype == bool:
            print(
                "ASSERTING Number type didn't change: ",
                _result_larry.dtype,
                larry.dtype,
            )
            assert _result_larry.dtype == larry.dtype

        ### Assert metadata
        print(
            "ASSERTING Metadata didn't change: ",
            "\n",
            _result_larry[0].metadata,
            "\n",
            larry[0].metadata,
        )
        assert _result_larry[0].metadata == larry[0].metadata

        ### Assert no conversion
        assert isinstance(_result_larry, l2Darrays)

        ### Assert correct numbers
        _result_ARRAY = _get_result_asarray(
            _operation,
            larry,
            other,
        )
        ### Assert Shape
        print("ASSERTING Shapes:", _result_larry.shape, _result_ARRAY.shape)
        assert _result_larry.shape == _result_ARRAY.shape

        ### Check similarity value by value
        print("ASSERTING Value by value:")
        print("=== larry result: ===\n", _result_larry)
        print("=== array result: ===\n", _result_ARRAY)
        _assert_value_by_value(_result_larry, _result_ARRAY)

        print("> TEST PASSED \n")

    print("> op: Addition (larry + other) ")
    _operation = lambda x1, x2: x1 + x2
    _result_larry = larry + other
    _assertions()

    print("> op: Subtraction (larry - other) ")
    _operation = lambda x1, x2: x1 - x2
    _result_larry = larry - other
    _assertions()

    print("> op: Multiplication (larry * other) ")
    _operation = lambda x1, x2: x1 * x2
    _result_larry = larry * other
    _assertions()

    print("> op: Division (larry / other) ")
    _operation = lambda x1, x2: x1 / x2
    _result_larry = larry / other
    _assertions()

    ### Comparisons
    print("> op: Greater (larry > other) ")
    _operation = lambda x1, x2: x1 > x2
    _result_larry = larry > other
    _assertions()

    print("> op: Greater Equal (larry >= other) ")
    _operation = lambda x1, x2: x1 >= x2
    _result_larry = larry >= other
    _assertions()

    print("> op: Less (larry < other) ")
    _operation = lambda x1, x2: x1 < x2
    _result_larry = larry < other
    _assertions()

    print("> op: Less Equal (larry <= other) ")
    _operation = lambda x1, x2: x1 <= x2
    _result_larry = larry <= other
    _assertions()

    print("> op: Equal (larry == other) ")
    _operation = lambda x1, x2: x1 == x2
    _result_larry = larry == other
    _assertions()

    print("> op: Not Equal (larry != other) ")
    _operation = lambda x1, x2: x1 != x2
    _result_larry = larry != other
    _assertions()

    ### Match dtypes for operations where type matters
    larry = larry.astype(np.float32)
    if not isinstance(other, (int, float)) and not np.isscalar(other):
        other = other.astype(np.float32)

    # > large numbers get out of range for float16
    print("> op: Power (larry ** other) ")
    _operation = lambda x1, x2: x1**x2
    _result_larry = larry**other
    _assertions()

    print("> op: Floor Division (larry // other) ")
    _operation = lambda x1, x2: x1 // x2
    _result_larry = larry // other
    _assertions()

    # > Modulo is also handled by int other than float
    print("> op: Modulo (larry % other) ")
    _operation = lambda x1, x2: x1 % x2
    _result_larry = larry % other
    _assertions()


def _test_list2Darrays_mathoperations_multiple(
    larries: list[l2Darrays],
    others: list[int | float | T.array | l2Darrays],
):
    for larry in larries:
        print("=> ======= TEST SET: =========")
        print("=> larry:", larry)
        print()
        for other in others:
            print("> NEW TEST:")
            print(f"> other:   {type(other)}\n", other)
            _test_list2Darrays_math_operations(larry, other)
        print()
        print()


def _get_result_asarray(
    op,
    larry: l2Darrays,
    other: int | float | T.array,
):
    """Converts larry to array and performs operation. If larry is
    inhomogenous, it iterates through elements and performs operations
    elementwise"""
    _larry = larry.copy()

    if _larry.homogenous:
        return op(_larry.asarray(), other)
    else:
        for i, arr in enumerate(_larry):
            if isinstance(other, (int, float)) or np.isscalar(other):
                _larry[i] = op(arr, other)
            elif isinstance(other, (T.array, l2Darrays)):
                if _larry[i].shape == other[i].shape:
                    _larry[i] = op(arr, other[i])
            else:
                raise ValueError(
                    f"Couldn't perform operation with {type(other)}"
                )
    return _larry


if __name__ == "__main__":
    ### TEST: Scalars with all larries
    others_scalars = [
        0,
        1,
        2,
        0.5,
        np.float16(0.5),
    ]
    _test_list2Darrays_mathoperations_multiple(
        larries=ed_larries.larries_s_all,
        others=others_scalars,
    )
    # %%
    ### TEST: Larries homo with themselves
    _test_list2Darrays_mathoperations_multiple(
        larries=ed_larries.larries_s_homo,
        others=ed_larries.larries_s_homo,
    )
    # %%
    ### TEST: larries Homo with Larries as arrays
    larries_s_homo_array = [
        larry.asarray() for larry in ed_larries.larries_s_homo
    ]
    _test_list2Darrays_mathoperations_multiple(
        larries=ed_larries.larries_s_homo,
        others=larries_s_homo_array,
    )

    # %%
    ### TEST: Larries hetero with each other
    for i in range(len(ed_larries.larries_s_all)):
        # others_hetero = [larry_hetero_s_total[i]]
        # others = [larry_hetero_s_total[i]]
        _test_list2Darrays_mathoperations_multiple(
            larries=[ed_larries.larries_s_all[i]],
            others=[ed_larries.larries_s_all[i]],
        )


# %%
# ======================================================================
# == Boolean Indexing ==================================================


def _test_list2Darrays_bool_indexing():
    """Tests selection with arrays of booleans"""
    arr = ed_larries.larry_homo_s.asarray()
    print("> Original Array")
    print(arr)

    larry = ed_larries.larry_homo_s.copy()
    print("> Original larry:")
    print(larry)

    print()

    # > Bool selection returns 1D array, like numpy
    print("> numpy arr after boolean indexing:")
    arr_bool = arr > 9
    print(arr_bool)

    print("> larry after boolean indexing")
    larry_bool = larry > 9
    print(larry_bool)
    assert _assert_value_by_value(larry_bool, arr_bool)
    print()

    # > Replacing  values by bool-indexing preserves shape
    print("> numpy arr after setting value by boolean indexing:")
    arr[arr > 9] = 123
    print(arr)

    larry[larry > 9] = 123
    print("> larry after setting value by boolean indexing:")
    print(larry)
    print()
    assert _assert_value_by_value(larry, arr)


if __name__ == "__main__":
    _test_list2Darrays_bool_indexing()

    # %%
    # == Get familiar with numpy indexing ===
    arr = ed_larries.larry_homo_s.asarray()
    print(arr.shape, arr.dtype, type(arr))
    print(arr.shape)
    print(arr)
    # %%
    # > Bool selection preserves shape
    arr > 9
    # %%
    # ? Bool indexing returns 1D Array
    arr[arr > 9]
    # %%
    # > Bool indexing with 1D array returns arrays
    arr[[True, False, True]]
    # %%
    arr[True].shape

    # %%
    arr[:].shape

    # %%

    # %%
    val = [True, False, True]
    val = np.array([True, False, True])
    [array[val[i]] for i, array in enumerate(arr)]
    # %%
    np.concatenate([array[val[i]] for i, array in enumerate(arr)])
    # %%
    # val = [True, False, True]
    val = arr > 9
    ääh = [array[val[i]] for i, array in enumerate(arr)]
    [array[0] for array in ääh if array.shape[0] != 0]

    # %%
    # > this implementation preserves z-dimension, (losing only y-dimension)
    val = arr > 9
    [_arra[val[i]] for i, _arra in enumerate(arr)]
    # %%
    # > Replacing values by bool index preserves shape
    arr[arr > 9] = 123
    arr
    # %%
    # == Now implement that with list2Darrays ==
    larry = ed_larries.larry_homo_s.copy()
    print(larry.shape, larry.dtype, type(arr))
    print(larry)
    # %%
    # ? Bool selection preserves shape
    larry > 9

    # %%
    val = [True, False, True]
    val = np.array([True, False, True])
    sliced = [array[val[i]] for i, array in enumerate(larry)]
    sliced
    #%%
    larry[val]
    
    
    # %%
    np.array([True, False, True])
    # %%
    # print(aa_c[0].name) # !! np.concat destroy metadata
    aa_c = np.concatenate(sliced)
    aa_c

    # %%
    aa_c2 = [array for array in sliced if array.shape[0] != 0]
    aa_c2

    # %%
    np.array(aa_c2).shape

    # %%
    def concatenate(sliced):
        ### Remove empty arrays
        return l2Darrays(
            [array.squeeze() for array in sliced if array.shape[0] != 0]
        )
    conc = concatenate(sliced)
    conc

    # %%
    print(sliced[0].name)  # ?? Does not lose metadata !!
    
    # %%
    sliced2 = larry > 9
    conc2 = concatenate(sliced2)
    conc2
    
    # %%
    # !! Bool indexing returns a LIST of 1D arrays
    larry[larry > 9]
    # %%
    # > this implementation preserves z-dimension, (losing only y-dimension)
    val = arr > 9
    [_arra[val[i]] for i, _arra in enumerate(larry)]
    # %%
    # > An implementation that mimics numpy behavior
    # > Since numpy intentionally loses dimension information, we might
    # ' also abandon metadata information
    val = arr > 9
    np.concatenate([_arra[val[i]] for i, _arra in enumerate(larry)])
    # %%

    # %%
    # > Replacing  values by bool-indexing preserves shape
    # larry[larry > 9] = 123
    # larry

    # %%
    np.atleast_2d(np.array([1, 2, 3]))


### Test as part of Collection
# %%
if __name__ == "__main__":
    import imagep as ip

    folder = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    Z = ip.Stack(
        data=folder,
        fname_extension="txt",
        verbose=True,
        pixel_length=(1.5 * 115.4) / 1024,
        imgname_position=1,
    )
    I = 6

    # %%
    loar = l2Darrays(arrays=list(Z.imgs))
    print(loar.shapes)
    print(loar.shapes[2])
    print(loar.shape)

    # %%
    # Z.imgs.tolist()
