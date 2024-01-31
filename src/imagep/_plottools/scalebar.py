#
# %%
import numpy as np

from PIL import Image, ImageDraw, ImageFont

# %%


def burn_scalebar_to_img(
    img: np.ndarray = None,
    pixel_size: float = None,
    microns: int = 10,
    thickness_px: int = 20,  # > In pixels
    xy_pad: tuple[float] = (0.05, 0.05),
    bar_color: int | float = None,
    frame_color: int | float = None,
) -> np.ndarray:
    """_summary_

    :param img: _description_, defaults to None
    :type img: np.ndarray, optional
    :param pixel_size: _description_, defaults to None
    :type pixel_size: float, optional
    :param microns: _description_, defaults to 10
    :type microns: int, optional
    :param xy_pad: Distance from bottom right corner in % of image size, defaults to (0.05, 0.05)
    :type xy_pad: tuple[float], optional
    :param bar_color: The bar color measured on the same scale as the
        image (0 - 255, or 0.0 - 1.0), defaults to None
    :type bar_color: int, optional
    :param frame_color: The bar frame color measured on the same scale as the
        image (0 - 255, or 0.0 - 1.0), defaults to None
    :type frame_color: int, optional
    :return: _description_
    :rtype: np.ndarray
    """
    bar_color = img.max() * 1 if bar_color is None else bar_color
    frame_color = img.max() * 0.9 if frame_color is None else frame_color

    ### Convert µm to pixels
    len_px = int(round(microns / pixel_size))
    # thickness_px = int(round(thickness / pixel_size))

    ### Define Scalebar as an array
    # > Color is derived from img colormap
    scalebar = np.zeros((thickness_px, len_px))
    scalebar[:, :] = bar_color

    ### Add Frame around scalebar with two pixels thickness
    t = 3  # Thickness of frame in pixels
    scalebar[0 : t + 1, :] = frame_color
    scalebar[-t:, :] = frame_color
    scalebar[:, 0 : t + 1] = frame_color
    scalebar[:, -t:] = frame_color

    ### Define padding from bottom right corner
    pad_x = int(img.shape[1] * xy_pad[0])
    pad_y = int(img.shape[0] * xy_pad[1])

    ### Burn scalebar to the bottom right of the image
    # !! Won't work if nan are at scalebar position
    img[-pad_y - thickness_px : -pad_y, -pad_x - len_px : -pad_x] = scalebar

    return img


def burn_micronlength_to_img(
    img: np.ndarray,
    microns: int = 10,
    thickness_px=3,
    xy_pad: tuple[float] = (0.05, 0.05),
    color: str | tuple[int] = None,
) -> np.ndarray:
    ### Convert into pil image
    dtype = img.dtype  # > Remember dtype
    # img = Image.fromarray((img * 255).astype(np.uint32))
    img = Image.fromarray(img, mode="F")  # > mode ="F" means float32

    ### Define Text
    text = f"{microns} µm"
    color = img.getextrema()[1] if color is None else color

    ### Define padding from bottom right corner
    pad_x = int(img.size[0] * xy_pad[0])
    pad_y = int(img.size[1] * xy_pad[1])

    ### Define position of text (in pixels)
    x = img.size[0] - pad_x - thickness_px * 2
    y = img.size[1] - pad_y - thickness_px

    ### Add text to image
    font = ImageFont.truetype("Arial Narrow.ttf", size=img.size[0] / 20)
    # font = ImageFont.load("arial.pil")
    pil_draw = ImageDraw.Draw(img)
    pil_draw.fontmode = "1"  # > "1" disables Anti-aliasing, "L" enables
    pil_draw.text(
        (x, y),
        text,
        fill=color,
        font=font,
        anchor="mb",  # > mb = middle bottom, ms = middle baseline
    )

    ### Convert back to numpy array
    # img = np.array(img, dtype=dtype) / 255
    img = np.array(img, dtype=dtype)
    return img


# def annot_micronlength_into_plot(
#     img: np.ndarray,
#     pixel_size: float,
#     microns: int = 10,
#     thickness=3,
#     xy_pad: tuple[float] = (0.05, 0.05),
#     color="white",
# ) -> np.ndarray:
#     """Adds length of scalebar to image as text during plotting"""

#     ### Define Text
#     text = f"{microns} µm"
#     # offsetbox = TextArea(text, minimumdescent=False)

#     ### Define padding from bottom right corner
#     pad_x = int(img.shape[1] * xy_pad[0])
#     pad_y = int(img.shape[0] * xy_pad[1])

#     ### Define position of text
#     x = img.shape[1] - pad_x - thickness / pixel_size * 2
#     y = img.shape[0] - pad_y - thickness / pixel_size

#     coords = "data"  # > Use array coordinates
#     plt.annotate(
#         text,
#         xy=(x, y),
#         xycoords=coords,
#         xytext=(x, y),
#         textcoords=coords,
#         ha="center",
#         va="bottom",
#         fontsize=10,
#         color=color,
#     )
