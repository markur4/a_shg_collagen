"""Stores instances of list2Darrays for testing"""

import numpy as np

# > Local
from imagep.images.list2Darrays import list2Darrays
from imagep.images.mdarray import mdarray


#
# == Larger Larries ====================================================

larry_homo = list2Darrays(
    arrays=[np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10))],
    dtype=np.float64,
)
larry_hetero_type = list2Darrays(
    arrays=[np.ones((10, 10)), np.ones((20, 20)), np.ones((30, 30))],
    dtype=np.float64,
)
larry_hetero_total = list2Darrays(
    arrays=[
        np.ones((10, 10), dtype=np.uint8),
        np.ones((20, 20), dtype=np.float64),
        np.ones((30, 30), dtype=np.float16),
    ],
)
larries = [
    larry_homo,
    larry_hetero_type,
    larry_hetero_total,
]
homo_shapes = [True, False, False]
homo_dtypes = [True, True, False]

args = zip(
    larries,
    homo_shapes,
    homo_dtypes,
)

#
# == Smaller Larries ===================================================
larry_homo_s = list2Darrays(
    arrays=[
        np.ones((2, 2)),
        np.ones((2, 2)),
        np.ones((2, 2)),
    ],
    dtype=np.float64,
)
larry_homo_s_heterotype = list2Darrays(
    arrays=[
        np.ones((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.float64),
        np.ones((2, 2), dtype=np.float16),
    ]
)
larry_hetero_s = list2Darrays(
    arrays=[np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4))],
    dtype=np.float64,
)
larry_hetero_s_total = list2Darrays(
    arrays=[
        np.ones((2, 2), dtype=np.uint8),
        np.ones((3, 3), dtype=np.float64),
        np.ones((4, 4), dtype=np.float16),
    ],
)

### Add metadata and multiply
for larry in [
    larry_homo_s,
    larry_homo_s_heterotype,
    larry_hetero_s,
    larry_hetero_s_total,
]:
    for z, img in enumerate(larry):
        larry[z] = mdarray(img, pixel_length=20, name=f"testImage {z}")
        # larry[z] = larry[z] * 3 * z + 50
        ### Make it even more4 uneven
        for y in range(larry.shape[1]):
            for x in range(larry.shape[2]):
                # print(z,y,x)
                img[y, x] = img[y, x] + (z+1) + (y+2) + (x+4) 
                

### Collect larries
larries_s_homo = [
    larry_homo_s,
    larry_homo_s_heterotype,
]

larries_s_all = [
    larry_homo_s,
    larry_homo_s_heterotype,
    larry_hetero_s,
    larry_hetero_s_total,
]
# %%
if __name__ == "__main__":
    for i in range(len(larries_s_homo)):
        print(larries_s_homo[i])
#%%