# src/easy_3dgs/pipeline/feature_matching/hloc_implementation.py
import logging
from pathlib import Path
from hloc import match_features, match_dense
from .base import AbstractFeatureMatcher, AbstractDenseFeatureMatcher


class HlocFeatureMatcher(AbstractFeatureMatcher):
    """Concrete implementation for feature matching using HLOC."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, pairs_path: Path, feature_output_name: str, output_dir: Path):
        logging.info("Step 4/6: Matching features...")
        return match_features.main(
            self.config, pairs_path, feature_output_name, output_dir
        )


class HlocDenseFeatureMatcher(AbstractDenseFeatureMatcher):
    """Concrete implementation for feature matching using HLOC."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, pairs_path: Path, image_dir: str, output_dir: Path):
        logging.info("Step 4/6: Matching features...")
        return match_dense.main(self.config, pairs_path, image_dir, output_dir)
