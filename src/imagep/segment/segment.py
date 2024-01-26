"""Module to Segment image into regions """

# %%
import os

import numpy as np
import matplotlib.pyplot as plt

### Increase display dpi of matplotlib
plt.rcParams["figure.dpi"] = 300
import seaborn as sns

import scipy as sp

import skimage as ski
from sklearn.ensemble import RandomForestClassifier

from imagep.preprocess.preprocess import PreProcess
# from imagep.visualise.zstack import ZStack

# %%
# ======================================================================


class Segment(PreProcess):
    def __init__(
        self,
        *imgs_args,
        mip_binning: bool | int = False,
        smoothen_edges_imgs: bool | str = "median",
        segment_method: str = "random_forest",
        # sigma: float = 4,
        open_segment: bool | int = False,
        smoothen_edges_segment: bool | str = "median",
        **preprocess_kws,
    ):
        super().__init__(*imgs_args, **preprocess_kws)

        ### Collect kws
        self.kws_segment = {
            "mip_binning": mip_binning,
            "smoothen_edges_imgs": smoothen_edges_imgs,
            "segment_method": segment_method,
            "open_segment": open_segment,
            "smoothen_edges_segment": smoothen_edges_segment,
            # "sigma": sigma,
        }

        ### History
        # TODO: Add history

        # t = self.imgs.dtype

        ### Intermediates of Segmentation
        self._imgs = self.imgs  # > Start with preprocessed images
        self._imgs_mipwin = np.zeros(self.imgs.shape, dtype=self.imgs.dtype)
        self._imgs_smooth = np.zeros(self.imgs.shape, dtype=self.imgs.dtype)
        self._segm_raw = np.zeros(self.imgs.shape, dtype=self.imgs.dtype)
        self._segm_open = np.zeros(self.imgs.shape, dtype=self.imgs.dtype)
        self._segm_smooth = np.zeros(self.imgs.shape, dtype=self.imgs.dtype)

        ### Execute Segmentation
        self.segmented: np.ndarray = self.segment_main(**self.kws_segment)

    def segment_main(self, **segment_kws) -> np.ndarray:
        """Main function to segment the images"""
        
        if self.verbose:
            print("=> Segmenting ...")
        
        ### Mip binning: Reduce z-resolution by binning?
        mip_bin = segment_kws["mip_binning"]
        if mip_bin:
            self._imgs_mipwin = self.mip_binning(self._imgs, windowsize=mip_bin)
            self._imgs = self._imgs_mipwin
        else:
            self._imgs_mipwin = self.mip_binning(self._imgs, windowsize=2)

        ### Smoothen edges before use?
        smo_img = segment_kws["smoothen_edges_imgs"]
        if smo_img:
            self._imgs_smooth = self.smoothen_edges(self._imgs, method=smo_img)
            self._imgs = self._imgs_smooth
        else:
            self._imgs_smooth = self.smoothen_edges(self._imgs, method="median")

        ### Segment
        self._segm_raw = self.segment(
            self._imgs,
            method=segment_kws["segment_method"],
        )
        self._imgs = self._segm_raw

        ### Smoothen edges after segmentation?
        smo_seg = segment_kws["smoothen_edges_segment"]
        if smo_seg:
            self._segm_smooth = self.smoothen_edges(self._imgs, method=smo_seg)
            self._imgs = self._segm_smooth

        ### Open morphological operation: erosion followed by dilation
        open_seg = segment_kws["open_segment"]
        if open_seg:
            self._segm_open = self._open_morph(self._imgs, iterations=open_seg)
            self._imgs = self._segm_open
        else:
            self._segm_open = self._open_morph(self._imgs, iterations=2)

        if self.verbose:
            print("   Segmentation done")

        return self._imgs

    # == MIP binning ===================================================

    @staticmethod
    def mip_binning(imgs: np.ndarray, windowsize: int = 2) -> np.ndarray:
        """performs maximum intensity projection along z-axis

        Advantages of MIP:
        - Higher Quality (less noise, stronger signal)
        - Thresholding on mip is robuster, higher background, ignores
        weak signals

        - Orientation of fibers now irrelevant
        - Less computation time

        Disadvantages:
        - If too many fibers, segmentation becomes impossible

        Overall, this turned out to be not so good
        """

        ### Get n images from stack with stepsize 1
        stack = np.zeros(imgs.shape, dtype=imgs.dtype)
        for i in range(imgs.shape[0] - windowsize):
            imgs_window = imgs[i : i + windowsize]

            ### MIP
            stack[i] = np.max(imgs_window, axis=0)

        return stack

    #
    # == Filters and Edits ================================================
    @staticmethod
    def smoothen_edges(imgs: np.ndarray, method: str = "median") -> np.ndarray:
        """Smoothen the edges of the images to reduce noise and artifacts
        from segmentation"""
        ### Define smoothening method
        if method == "median":
            imgs_smoothened = Segment._smoothen_median(imgs)
        # elif method == "guided":
        #     pass
        #     # smoothened = filters.mean(S)
        else:
            raise ValueError(f"Smoothening method {method} not implemented")

        return imgs_smoothened

    @staticmethod
    def _smoothen_median(imgs: np.ndarray, kernel_size=5) -> np.ndarray:
        """Smoothen the edges of the images to reduce noise and artifacts
        from segmentation"""
        imgs_smooth = np.zeros(imgs.shape).astype(imgs.dtype)
        for i, img in enumerate(imgs):
            imgs_smooth[i] = sp.signal.medfilt(img, kernel_size=kernel_size)

        return imgs_smooth

    @staticmethod
    def _open_morph(imgs: np.ndarray, iterations=2) -> np.ndarray:
        """Open morphological operation: erosion followed by dilation"""
        R = np.zeros(imgs.shape).astype(np.uint8)
        for i, img in enumerate(imgs):
            R[i] = sp.ndimage.binary_opening(img, iterations=iterations)

        return R

    #
    # == Thresholding Methods ==========================================

    @staticmethod
    def segment(imgs: np.ndarray, method: str = "random_forest") -> np.ndarray:
        """Performs different segmentation methods"""
        if method == "random_forest":
            return Segment._segment_random_forest()
        elif method == "background":
            return Segment._segment_background(imgs, thresh_per_img=True)
        else:
            raise ValueError(f"Segmentation method {method} not implemented")

    @staticmethod
    def _segment_background(
        imgs: np.ndarray, thresh_per_img=True, sigma=2
    ) -> np.ndarray:
        """Segments by thresholding the background"""

        ### Init empty stack
        R = np.zeros(imgs.shape).astype(np.uint8)

        ### Segmentation
        for i, img in enumerate(imgs):
            ### Recalculate threshold for each individual image
            if thresh_per_img:
                img_blur = ski.filters.gaussian(img, sigma=sigma)
                threshold = ski.filters.threshold_triangle(img_blur)
            else:
                threshold = 0

            ### Segment
            R[i] = img > threshold
            R[i] = R[i].astype(np.uint8)
        return R

    def _segment_random_forest(imgs: np.ndarray) -> np.ndarray:
        pass

    #
    # == Feature Extraction ============================================
    
    #
    # == Visualization =================================================
    
    def plot_masked_by_segmentation(self, I=0, alpha=0.4):
        """Plot an image masked by segmentation"""
        img = self._imgs_use[I]
        seg = self.segmented[I]

        # > Mask
        masked = np.ma.masked_where(seg == 0, img)

        plt.imshow(masked, interpolation="none", cmap="gray", alpha=alpha)
    

# !!
# ======================================================================

# %%
if __name__ == "__main__":
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"

    Z = Segment(
        path=path,
        denoise=True,
        subtract_bg=True,
        # scalebar_micrometer=10,
        # mip_binninbg=2,
        smoothen_edges_imgs="median",
        segment_method="background",
        # open_segment=2,
        smoothen_edges_segment="median",
    )
    # %%
    Z.info

    # %%
    I = 0

    # %%
    # == Check Smoothening of image ====================================
    print(Z.imgs.dtype)
    plt.imshow(Z.imgs[I])
    # %%
    print(Z._imgs_smooth.dtype)
    plt.imshow(Z._imgs_smooth[I])

    # %%
    # == Check Segmentation ============================================
    print(Z._segm_raw.dtype)
    Z._segm_raw
    # %%
    plt.imshow(Z._segm_raw[I])
    # %%
    print(Z._segm_open.dtype)
    plt.imshow(Z._segm_open[I])
    # %%
    print(Z._segm_smooth.dtype)
    plt.imshow(Z._segm_smooth[I])

    # %%
    # == End Result Segmentation =======================================
    print(Z.segmented.dtype)
    plt.imshow(Z.segmented[I])

    # %%
    # == Visualize Segmentation ========================================
    Z.plot_masked_by_segmentation(I=0, alpha=0.4)

    # %%
    # == Test Feature Extraction =======================================
    print("sigma=1.0")
    F_canny_1 = ski.feature.canny(Z[I], sigma=1.0)
    plt.imshow(F_canny_1)

    # %%
    print("sigma=1.5")
    F_canny_1 = ski.feature.canny(Z[I], sigma=1.5)
    plt.imshow(F_canny_1)

    # %%
    print("sigma=2")
    F_canny_1 = ski.feature.canny(Z[I], sigma=2)
    plt.imshow(F_canny_1)

    # %%
    hää

    # %% ===============================================================
    # == Threshold with Machine Learning ===============================

    # %%
    ### Extract Features

    # from skimage import filters

    def make_feature_stack(image, sigma=1):
        ### Define features
        blurred = ski.filters.gaussian(image, sigma=sigma)
        # > Edge Detection
        prewitt = ski.filters.prewitt(blurred)
        sobel = ski.filters.sobel(blurred)
        canny1 = ski.feature.canny(blurred, sigma=1)  # > Detailed edges
        canny2 = ski.feature.canny(blurred, sigma=2)  # > Rough edges

        ### Collect features in a stack
        # > The ravel() function turns a nD image into a 1-D image.
        #  > We need to use it because scikit-learn expects values in a 1-D format here.
        feature_stack: list[np.ndarray] = [
            image.ravel(),
            blurred.ravel(),
            prewitt.ravel(),
            sobel.ravel(),
            canny1.ravel(),
            canny2.ravel(),
        ]

        return np.asarray(feature_stack)

    feature_stack = make_feature_stack(F)

    # %%

    ### show feature images
    import matplotlib.pyplot as plt

    names = [
        "Image",
        "blurred",
        "prewitt",
        "sobel",
        "canny\nsigma=1",
        "canny\nsigma=2",
    ]

    h = 2
    w = int(feature_stack.shape[0] / h)
    fig, axes = plt.subplots(h, w, figsize=(w * 5, h * 5))
    plt.subplots_adjust(wspace=0.03, hspace=0.2)

    for ax, img, name in zip(axes.flatten(), feature_stack, names):
        ax.imshow(img.reshape(F.shape), cmap=plt.cm.gray)
        ax.axis("off")
        ax.set_title(name)

    # %%
    def mask_by_annotation(feature_stack: np.ndarray, annotation: np.ndarray):
        """Reformat the data to match what scikit-learn expects"""
        ### Transpose the feature stack
        X = feature_stack.T
        ### Make the annotation 1-dimensional
        y = annotation.ravel()

        ### Remove all pixels from the feature and annotations which have not been annotated
        mask = y > 0
        X = X[mask]
        y = y[mask]

        return X, y

    # # %%
    # ### Pseudo-annotation
    # annotation = np.zeros(F.shape)
    # annotation[0:10, 0:10] = 1
    # annotation[45:55, 10:20] = 2

    # plt.imshow(annotation)

    # X, y = apply_annotation(feature_stack, annotation)

    # print("input shape", X.shape)
    # print("annotation shape", y.shape)

    # %%
    ### TRAIN
    # _classifier = RandomForestClassifier(max_depth=2, random_state=0)
    # _classifier.fit(X, y)

    # %%
    # ### PREDICT
    # # > we subtract 1 to make background = 0
    # res = _classifier.predict(feature_stack.T) - 1

    # plt.imshow(F)
    # plt.show()
    # plt.imshow(res.reshape(F.shape))
    # plt.show()
