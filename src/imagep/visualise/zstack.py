#
# %%

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 250

import vtk
from vtk.util import numpy_support
import pyvista as pv

# > Local
from imagep.preprocess.preprocess import PreProcess
import imagep.utils.utils as ut

# %%


class ZStack(PreProcess):
    def __init__(
        self,
        z_dist,
        scalebar: bool = True,
        *imgs_args,
        **preprocess_kws,
    ):
        """Visualize Z-stacks.
        :param z_dist: Distance between two images in µm
        :type z_dist: float
        :param scalebar: Switches on scalebar, defaults
            to True
        :type scalebar: bool, optional
        :param imgs_args: positional arguments passed to Imgs class
        :type imgs_args: tuple
        :param preprocess_kws: Keyword arguments passed to
        PreProcess class
        """
        super().__init__(*imgs_args, **preprocess_kws)

        ### Burn Scalebar into the first image:
        self.scalebar = scalebar
        if scalebar:
            self.burn_scalebar(all=False, indexes=[0])

        ### Z-distance
        self.z_dist = z_dist

        self.spacing = (self.x_µm, self.y_µm)

        ### Fill z
        self.stack_zfilled = self.fill_z()

    # ==================================================================
    # == Utils =========================================================

    def fill_z(self) -> np.ndarray:
        """Each pixel has a length in µm. Consider each image as a pixel
        in z-direction. This function duplicates images so that each
        image has a thickness of z_dist in µm"""

        ### Calculate number how often each image has to be repeated
        thickness_pixel = self.z_dist / self.pixel_size
        thickness_pixel = int(round(thickness_pixel))

        ### Duplicate images n times so that each layer has thickness
        # ' of µm
        stack = []
        for img in self.imgs:
            for _ in range(thickness_pixel):
                stack.append(img)

        ### Convert to numpy array
        stack = np.array(stack)
        return stack

    # == __str__ ======================================================

    @property
    def _info_ZStack(self):
        adj = self._adj
        ID = self._info

        ### Append Z Information to "Distance"
        ID["Distance"] = ID["Distance"] + [
            adj("pixel size z") + f"{self.z_dist}",
            adj("z") + f"{self.z_dist * self.imgs.shape[0]}",
        ]

        return ID

    def __str__(self) -> str:
        return self._info_to_str(self._info_ZStack)

    # ==================================================================
    # == Plotting ======================================================

    def mip(self, **mip_kws) -> np.ndarray | None:
        """Maximum intensity projection across certain axis"""
        #!! Overrides PreProcess.mip(), so that correct z-axis dimension
        #!! is used

        ### We need an img to add micronlength, but don't duplicate
        return_array: bool = mip_kws.pop("return_array", False)

        ### Mip
        mip = ut.mip(self.stack_zfilled, return_array=True, **mip_kws)

        ### Annotate length of scalebar in µm
        if self.scalebar and mip_kws.get("axis", 0) == 0:
            self.annot_micronlength_into_plot(img=mip)

        ### Return if initially requested (was popped out of mip_kws)
        if return_array:
            return mip

    @property
    def stack_vtk(self):
        stack = self.stack_zfilled

        ### Transpose the numpy array to match PyVista's x, y, z convention
        stack = np.transpose(stack, axes=[2, 1, 0])

        ### Convert numpy array to VTK array
        vtk_data_array = numpy_support.numpy_to_vtk(
            num_array=stack.ravel(order="F"),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )

        ### Create vtkImageData object
        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(stack.shape)
        vtk_image_data.GetPointData().SetScalars(vtk_data_array)

        return vtk_image_data

    @property
    def stack_pyvista(self):
        return pv.ImageData(self.stack_vtk)

    def plot_volume(
        self,
        cmap="viridis",
        show=True,
        ret=False,
        **plot_kws,
    ) -> pv.Plotter:
        """Makes a Plotter object with the volume added"""
        ### Plot
        pl = pv.Plotter()

        ### Add volume
        pl.add_volume(
            self.stack_pyvista,
            cmap=cmap,
            # opacity="sigmoid",
            **plot_kws,
        )

        ### Edits
        # > Bounding Box
        pl.add_bounding_box(color="#00000050")  # > transparent black
        # > Camera Position
        pl.camera.roll = 180

        ### Return
        if show:
            pl.show()
        if ret:
            return pl

    def makegif_rotate(
        self,
        fname: str = "rotate.gif",
        angle_per_frame=1,
    ):
        ### Initialize plotter
        pl = self.plot_volume(show=False, ret=True)

        ### Set initial camera position
        pl.camera_position = "xy"
        pl.camera.roll = 180
        # pl.camera.roll = 45

        ### Adjust clipping range
        # pl.camera.clipping_range = self.stack.shape[1:2]
        pl.camera.clipping_range = (1000, 5000)

        ### Open gif
        if not fname.endswith(".gif"):
            fname += ".gif"
        pl.open_gif(fname)

        ### Rotate & Write
        for angle in range(0, 360, angle_per_frame):
            ### adjust zoom to keep object in frame

            pl.camera.azimuth = angle
            pl.render()
            pl.write_frame()

        pl.close()


if __name__ == "__main__":
    pass
    # %%
    ### Import from a txt file.
    # > Rough
    path = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/1 healthy z-stack rough/"
    kws = dict(
        z_dist=10 * 0.250,  # stepsize * 0.250 µm
        x_µm=1.5 * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    )

    # %%
    Z = ZStack(
        path=path,
        denoise=False,
        subtract_bg=True,
        **kws,
    )
    Z

    # %%
    Z.mip(axis=0)
    Z.mip(axis=1)
    Z.mip(axis=2)

    # %%
    ### make gif with denoised!
    Z2 = ZStack(
        path=path,
        denoise=True,
        subtract_bg=True,
        **kws,
    )
    Z2
    # %%
    Z2.mip()
    # %%
    # Z2.plot_volume()

    # %%
    Z2.makegif_rotate(fname="rotate_rough_scalebar=10", angle_per_frame=3)

    # %%
    ### make gif for detailed!
    # > Detailed
    path_detailed = "/Users/martinkuric/_REPOS/ImageP/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    kws_detailed = dict(
        z_dist=2 * 0.250,  # stepsize * 0.250 µm
        x_µm=1.5 * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    )

    Z_d = ZStack(
        path=path_detailed,
        denoise=True,
        subtract_bg=True,
        **kws_detailed,
    )
    Z_d
    # %%
    Z_d.makegif_rotate(fname="rotate_detailed_scalebar=10", angle_per_frame=3)
