import os
import shutil
import logging

from .base import BaseResizer

class ImageMagickResizer(BaseResizer):
    def main(self, sfm_dir: str, magnifications: list[int], magick_command: str = ""):
        print("Copying and resizing...")
        
        image_dir = os.path.join(sfm_dir, "images")
        files = os.listdir(image_dir)

        for mag in magnifications:
            if mag == 1:
                continue
                
            output_dir = os.path.join(sfm_dir, f"images_{mag}")
            os.makedirs(output_dir, exist_ok=True)

            for file in files:
                source_file = os.path.join(image_dir, file)
                destination_file = os.path.join(output_dir, file)
                shutil.copy2(source_file, destination_file)
                
                resize_percentage = 100 / mag
                exit_code = os.system(f"{magick_command} mogrify -resize {resize_percentage}% {destination_file}")
                
                if exit_code != 0:
                    logging.error(f"{resize_percentage}% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)
