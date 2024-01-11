#
# %%

import numpy as np

import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support
import pyvista as pv

from mkshg.import_and_preprocess import PreProcess

# %%


class ZStack(PreProcess):
    def __init__(self, z_dist, **preprocess_kwargs,):
        super().__init__(**preprocess_kwargs)
        
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
        ### of thickness µm
        stack = []
        for img in self.stack:
            for _ in range(thickness_pixel):
                stack.append(img)

        ### Convert to numpy array
        stack = np.array(stack)
        return stack

    # ==================================================================
    # == Plotting ======================================================

    def mip(
        self,
        axis: int = 0,
        show=True,
        ret=False,
        savefig: str = "mip.png",
        colormap: str = "gist_ncar",
    ) -> np.ndarray:
        """Maximum intensity projection across certain axis"""
        #!! Overrides PreProcess.mip()
        mip = self.stack_zfilled.max(axis=axis)

        if show:
            plt.imshow(mip, cmap=colormap)
            plt.show()

        if savefig:
            plt.imsave(fname=savefig, arr=mip, cmap=colormap)
            
        if ret:
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
    # path =
    # "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/1
    # healthy z-stack rough/"
    # kws = dict(
    #     z_dist=10 * 0.250,  # stepsize * 0.250 µm
    #     x_µm=1.5 * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    # )
    # > Detailed
    path = "/Users/martinkuric/_REPOS/a_shg_collagen/ANALYSES/data/231215_adipose_tissue/2 healthy z-stack detailed/"
    kws = dict(
        z_dist=2 * 0.250,  # stepsize * 0.250 µm
        x_µm=1.5 * 115.4,  # fast axis amplitude 1.5 V * calibration 115.4 µm/V
    )
    # %%
    Z = ZStack(path=path, **kws)
    Z.stack.shape
    Z.mip()

    # %%
    Z2 = ZStack(
        path=path,
        denoise=True,
        background_subtract=0.06,
        scalebar_μm=20,
        **kws,
    )
    # %%
    Z2.mip()
    # %%
    Z2.mip(axis=2)
    # %%
    Z2.plot_volume()

    # %%
    # Z2.makegif_rotate(angle_per_frame=3)

# %%
