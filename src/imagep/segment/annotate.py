#
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import napari



# %%
### == MANUAL ANNOTATION ===============================================
if __name__ == "__main__":

    

    ### Open Viewer and ADd raw image
    viewer = napari.Viewer()

    # %%
    viewer.add_image(F, name="raw")

    # %%
    ### Add an empty labels layer and keet it in a variable
    labels = viewer.add_labels(np.zeros(F.shape).astype(int), name="labels")

    # %%
    ### Show annotations
    manual_annotations = labels.data
    plt.imshow(manual_annotations, vmin=0, vmax=2)

    # %%
    def open_annotations(
        fname="_annot_cache.npy",
    ) -> None | np.ndarray:
        """Load the annotations from the hard drive and display them in the viewer"""
        if not os.path.exists(fname):
            print("No annotations found")
            return
        with open(fname, "rb") as f:
            A = np.load(f)

        return A

    plt.imshow(open_annotations())

    # %%
    def update_annotations(
        viewer: napari.Viewer = None,
        fname="_annot_cache.npy",
    ):
        """Cache the current annotations in the viewer onto the hard
        drive and load them again when the viewer is opened again"""

        ### no viewer
        if viewer is None:
            print(f"No viewer, opening {fname}")
            return open_annotations(fname)

        ### File not existing
        if not os.path.exists(fname):
            print(f"Saving annotations to {fname}")
            np.save(fname, viewer.layers["labels"].data)
            A_viewer = viewer.layers["labels"].data

        ### File exists
        else:
            A_hd: np.ndarray = open_annotations(fname)
            A_viewer: np.ndarray = viewer.layers["labels"].data

            ### Viewer empty
            if A_viewer is None:
                print("Adding saved annotations to viewer")
                # > Display the annotations in the current viewer
                viewer.layers["labels"].data = A_hd
                A_viewer = A_hd

            ### Viewer matches the cached annotations
            elif np.array_equal(A_hd, A_viewer):
                print("Nothing done: Saved annotations match those in viewer")

            ### Viewer not matching the saved annotations
            elif not np.array_equal(A_hd, A_viewer):
                print("Updating annotations from viewer to hard drive")
                ### hd to viewer
                A_viewer[A_hd > 0] = A_hd[A_hd > 0]

                ### viewer to hd
                # A_hd[A_viewer > 0] = A_viewer[A_viewer > 0]

                # > Display the annotations in the current viewer
                viewer.layers["labels"].data = A_viewer

                np.save(fname, A_viewer)
        return A_viewer

    manual_annotations = update_annotations(
        viewer=viewer,
        fname="_annot_cache.npy",
    )
    plt.imshow(manual_annotations)

    # %%
    ### Train and predict
    # generate features (that's actually not necessary,
    # as the variable is still there and the image is the same.
    # but we do it for completeness)
    feature_stack = make_feature_stack(F)
    X, y = mask_by_annotation(feature_stack, manual_annotations)

    # train classifier
    classifier = RandomForestClassifier(max_depth=3, random_state=0)
    classifier.fit(X, y)

    # process the whole image and show result
    result_1d = classifier.predict(feature_stack.T)
    result_2d = result_1d.reshape(F.shape)
    plt.imshow(
        result_2d,
        # cmap=("red", "blue", "purple"),
    )

    # %%
    plt.imshow(F)

    # %%
    ### compare with non AI segmentation filtering
    # filter_image(F_blur, method="mean")
