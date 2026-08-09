import numpy as np
from PIL import Image


def preprocess_rgba_image(image: Image.Image, image_size: int = 518) -> Image.Image:
    """Apply TRELLIS pipeline cropping to an RGBA conditioning render."""
    if image.mode != "RGBA":
        raise ValueError("preprocess_rgba_image expects an RGBA image")

    image_np = np.asarray(image)
    foreground = np.argwhere(image_np[:, :, 3] > 0.8 * 255)
    if foreground.size == 0:
        raise ValueError("Conditioning image has an empty alpha mask")

    left = foreground[:, 1].min()
    top = foreground[:, 0].min()
    right = foreground[:, 1].max()
    bottom = foreground[:, 0].max()
    center = ((left + right) / 2, (top + bottom) / 2)
    size = int(max(right - left, bottom - top) * 1.2)
    if size <= 0:
        raise ValueError("Conditioning image foreground is too small")

    crop = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    image = image.crop(crop).resize((image_size, image_size), Image.Resampling.LANCZOS)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    rgb = image_np[:, :, :3] * image_np[:, :, 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8))
