"""Functions for extracting and applying metadata to images"""

# %%
import numpy as np

# > Local
from imagep.images.mdarray import mdarray
from imagep.images.l2Darrays import l2Darrays
import imagep._example_data.larries as edl
import imagep._utils.types as T


# %%
# == Extract and apply metadata =======================================


def extract_metadata(imgs: T.array | l2Darrays) -> list[dict]:
    """Extracts metadata from images and returns a list of
    dictionaries with key and value pairs"""
    if isinstance(imgs, mdarray):
        return imgs.metadata
    elif isinstance(imgs, l2Darrays):
        return [img.metadata for img in imgs]
    else:
        raise ValueError(f"imgs must be mdarray or l2Darrays, not {type(imgs)}")


def apply_metadata(
    imgs: np.ndarray,
    metadata: list[dict],
) -> l2Darrays:
    """Applies metadata to images"""
    if imgs.ndim == 2:
        # > Convert to enable metadata
        imgs = mdarray(imgs)
        for k, v in metadata.items():
            setattr(imgs, k, v)
        return imgs
    else:
        # > Convert to enable metadata
        imgs = l2Darrays([mdarray(img) for img in imgs])
        for img, md in zip(imgs, metadata):
            for k, v in md.items():
                setattr(img, k, v)

    return imgs


already_called = False


def preserve_metadata():
    """A decorator that wraps functions using images as first argument.
    It checks if the images have metadata, and if so, it extracts them
    before the funciton executes and then re-applies them after the
    function has executed.
    """

    def decorator(func):
        global already_called
        if already_called:
            # todo: This might not be necessary
            raise ValueError(
                "Decorator already called, no nested decorators allowed."
            )
        already_called = True

        def wrapper(*args, **kws):
            ### Get imgs from args
            if len(args) != 0:
                __imgs = args[0]
            elif "imgs" in kws:
                __imgs = kws["imgs"]
            elif "img" in kws:
                __imgs = kws["img"]
            else:
                raise ValueError("No argument 'imgs' found")

            ### Check Type
            if not isinstance(__imgs, (mdarray, l2Darrays)):
                raise ValueError(
                    f"Can't preserve metadata from type '{type(__imgs)}'"
                )
            if isinstance(__imgs, l2Darrays) and not all(
                isinstance(img, mdarray) for img in __imgs
            ):
                raise ValueError(
                    "Can't preserve metadata from l2Darrays with non-mdarray"
                )

            ### Execute function
            metadata = extract_metadata(__imgs)
            __imgs = func(*args, **kws)
            __imgs = apply_metadata(__imgs, metadata)
            return __imgs

        already_called = False
        return wrapper

    return decorator


if __name__ == "__main__":
    ### Get testdata
    larries = edl.larries_s_all
    print(f"{larries[0].arrays[2].metadata=}")

    # %%
    # > Test function
    @preserve_metadata()
    def test_func(imgs: l2Darrays, **kws):
        print("Test function executed")
        _imgs = np.array(imgs + 1)
        _imgs = l2Darrays(_imgs, dtype=imgs.dtype)
        return _imgs

    ### Test
    _new_larry = test_func(larries[0])
    # _new_larry
    print(f"{_new_larry.arrays[2].metadata=}")
