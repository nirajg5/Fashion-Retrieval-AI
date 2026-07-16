"""
Image Utilities
"""

from pathlib import Path
from PIL import Image

from config import IMAGE_DIR


class ImageLoader:

    def __init__(self):

        self.image_dir = IMAGE_DIR

    def load(self, image_path):

        return Image.open(image_path).convert("RGB")

    def exists(self, image_path):

        return Path(image_path).exists()

    def get_path(self, filename):

        return str(

            self.image_dir / filename

        )


image_loader = ImageLoader()