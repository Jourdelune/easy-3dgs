# src/easy_3dgs/pipeline/feature_matching/hloc_implementation.py
import logging
from pathlib import Path
from hloc import match_features
from .base import AbstractFeatureMatcher

class HlocFeatureMatcher(AbstractFeatureMatcher):
    """Concrete implementation for feature matching using HLOC."""
    def __init__(self, config: dict):
        self.config = config

    def run(self, pairs_path: Path, feature_output_name: str, output_dir: Path):
        logging.info("Step 4/6: Matching features...")
        return match_features.main(self.config, pairs_path, feature_output_name, output_dir)
