import imageio
from PIL import Image

def make_crop_gif(thumb_path, crop_path, out_path, duration=1.0):
    """Create a GIF cycling between thumbnail and crop."""
    thumb = Image.open(thumb_path).convert("RGB")
    crop = Image.open(crop_path).convert("RGB")
    thumb = thumb.resize(crop.size)
    imageio.mimsave(out_path, [thumb, crop], duration=duration)
    print(f"🎞️  Saved crop GIF: {out_path}")
