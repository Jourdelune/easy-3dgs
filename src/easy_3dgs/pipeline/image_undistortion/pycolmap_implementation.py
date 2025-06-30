# src/easy_3dgs/pipeline/image_undistortion/pycolmap_implementation.py
import logging
from pathlib import Path
import pycolmap
from .base import AbstractImageUndistorter

class PycolmapImageUndistorter(AbstractImageUndistorter):
    """Concrete implementation for image undistortion using Pycolmap."""
    def run(self, sfm_dir: Path, image_dir: Path):
        logging.info("Step 6/6: Undistorting images...")
        pycolmap.undistort_images(output_path=sfm_dir, image_path=image_dir, input_path=sfm_dir)
        logging.info("Image undistortion completed.")
