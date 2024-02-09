#
# %%
import numpy as np

from PIL import Image, ImageDraw, ImageFont

# > local imports
import imagep._utils.utils as ut


# %%
def burn_scalebars(
    imgs: np.ndarray,
    pixel_length: float,
    length: int = 10,
    thickness_px: int = 20,  # > In pixels
    xy_pad: tuple[float] = (0.05, 0.05),
    bar_color: int | float = None,
    frame_color: int | float = None,
    text_color: tuple[int] | str = None,
    inplace=False,
):

    imgs = imgs if inplace else imgs.copy()

    for i, img in enumerate(imgs):
        imgs[i] = burn_scalebar_to_img(
            img=img,
            length=length,
            pixel_length=pixel_length,
            thickness_px=thickness_px,
            xy_pad=xy_pad,
            bar_color=bar_color,
            frame_color=frame_color,
        )
        imgs[i] = burn_micronlength_to_img(
            img=img,
            length=length,
            thickness_px=thickness_px,
            xy_pad=xy_pad,
            textcolor=text_color,
        )
    return imgs


#


def burn_scalebar_to_img(
    img: np.ndarray,
    pixel_length: float,
    length: int = 10,
    thickness_px: int = 20,  # > In pixels
    xy_pad: tuple[float] = (0.05, 0.05),
    bar_color: int | float = None,
    frame_color: int | float = None,
) -> np.ndarray:
    """_summary_

    :param img: _description_, defaults to None
    :type img: np.ndarray, optional
    :param pixel_length: _description_, defaults to None
    :type pixel_length: float, optional
    :param length: _description_, defaults to 10
    :type length: int, optional
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
    len_px = int(round(length / pixel_length))
    # thickness_px = int(round(thickness / pixel_length))

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
    length: int = 10,
    thickness_px: int = 3,
    xy_pad: tuple[float] = (0.05, 0.05),
    textcolor: tuple[int] | str = None,
) -> np.ndarray:
    ### Convert into pil image
    dtype = img.dtype  # > Remember dtype
    # img = Image.fromarray((img * 255).astype(np.uint32))
    img = Image.fromarray(img, mode="F")  # > mode ="F" means float32

    ### Define Text
    text = f"{length} µm"
    textcolor = img.getextrema()[1] if textcolor is None else textcolor

    ### Define padding from bottom right corner
    pad_x = int(img.size[0] * xy_pad[0])
    pad_y = int(img.size[1] * xy_pad[1])

    ### Define position of text (in pixels)
    x = img.size[0] - pad_x
    y = img.size[1] - pad_y - thickness_px

    ### Define font
    font = ImageFont.truetype("Arial Narrow.ttf", size=img.size[0] / 20)
    # font = ImageFont.load("arial.pil")
    ### Anchor
    # > m = middle, r = right, l = left
    anchor_x = "r"
    # > s = baseline (lowest pixel), d = descender (lowest by font)
    anchor_y = "d"

    ### Add text to image
    # > Initialize drawing
    pil_draw = ImageDraw.Draw(img)
    pil_draw.fontmode = "1"  # > "1" disables Anti-aliasing, "L" enables
    pil_draw.text(
        (x, y), text, fill=textcolor, font=font, anchor=anchor_x + anchor_y
    )

    ### Convert back to numpy array
    # img = np.array(img, dtype=dtype) / 255
    img = np.array(img, dtype=dtype)
    return img
