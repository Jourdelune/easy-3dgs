# src/easy_3dgs/pipeline/reconstruction/hloc_implementation.py
import logging
from pathlib import Path
from hloc import reconstruction
from .base import AbstractReconstructor

class HlocReconstructor(AbstractReconstructor):
    """Concrete implementation for 3D reconstruction using HLOC."""
    def run(self, sfm_dir: Path, image_dir: Path, pairs_path: Path, feature_path: Path, match_path: Path, mapper_options: dict = None):
        logging.info("Step 5/6: Starting 3D reconstruction...")
        model = reconstruction.main(
            sfm_dir,
            image_dir,
            pairs_path,
            feature_path,
            match_path,
            mapper_options=mapper_options or {},
        )
        return model
