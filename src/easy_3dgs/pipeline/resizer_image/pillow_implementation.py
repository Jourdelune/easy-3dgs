import os
import logging
from PIL import Image

from .base import BaseResizer


class PillowResizer(BaseResizer):
    def main(self, sfm_dir: str, magnifications: list[int]):
        print("Copying and resizing with Pillow...")

        image_dir = os.path.join(sfm_dir, "images")
        if not os.path.isdir(image_dir):
            logging.error(f"Image directory not found at {image_dir}. Skipping resize.")
            return

        files = os.listdir(image_dir)

        for mag in magnifications:
            if mag == 1:
                continue

            output_dir = os.path.join(sfm_dir, f"images_{mag}")
            os.makedirs(output_dir, exist_ok=True)

            for file in files:
                source_file = os.path.join(image_dir, file)
                destination_file = os.path.join(output_dir, file)

                try:
                    with Image.open(source_file) as img:
                        width, height = img.size
                        new_width = int(width / mag)
                        new_height = int(height / mag)
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                        resized_img.save(destination_file)
                except Exception as e:
                    logging.error(f"Failed to resize {file} to 1/{mag}x: {e}")
