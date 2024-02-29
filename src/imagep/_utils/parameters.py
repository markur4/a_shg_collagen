"""Here, we collect parameters to be re-used by different functions.
Each parameter is a dictionary that assigns
"""

# %%
import numpy as np

# > Local
import imagep._utils.utils as ut


# %%
# == utils =============================================================
def docs_to_str(docs: dict[str, str]) -> str:
    """Converts object to string"""
    just = lambda x: ut.justify_str(x, justify=12)

    lines = []
    for key, val in docs.items():
        key = f"'{str(key)}'" if isinstance(key, str) else f"{str(key)}"
        key = f"- {key}"
        line = just(key) + val
        lines.append(line)

    return "\n".join(lines)
    # return "asdfasfd"


def handle_param(
    paramconfig: dict[dict],
    param: object,
    funcname: str = "",
) -> object:
    """Handles parameter
    - Checks if passed parameter is in the docs
    - Checks if an alternative parameter was passed
    - Raises ValueError if parameter is not recognized, and prints docs
    """
    docs = paramconfig["docs"]
    funcname = funcname + ": " if funcname else ""

    if param in paramconfig["alts"]:
        # print(f"param in alts: {param=}")
        return paramconfig["alts"][param]
    else:
        raise ValueError(
            f"{funcname}Argument '{param}' not recognized,"
            f" please pick:\n{docs_to_str(docs)}"
        )


# %%
# == Perform across image or stack ====================================
# Control if action is performed for each image, or across stack

# > Assign alternative inputs to "img" and "stack"

ACROSS = dict(
    docs={
        "img": "Applies operation for each image (ignoring the stack)",
        "stack": "Applies operation on complete stack, (e.g. calculates number for all images and using that number on every image)",
        # True: "Alias for 'image'",
        False: "Does not perform operation",
    },
    alts={
        False: False,
        # True: "img", # !! for normalize and across, this can be ambiguous
        "img": "img",
        "per_img": "img",
        "stack": "stack",
        "per_stack": "stack",
    },
)


if __name__ == "__main__":
    # pass
    vals = [False, "img", "stack", "per_img", "per_stack", "asdf", True]
    funcname = "blafunc"
    for val in vals:
        try:
            print(val, handle_param(ACROSS, param=val, funcname=funcname))
        except ValueError as e:
            # print(f"ERROR: Value '{val}' not recognized")
            print(e)
            
    #%%
    True in ACROSS["alts"]
    True in ACROSS["alts"]