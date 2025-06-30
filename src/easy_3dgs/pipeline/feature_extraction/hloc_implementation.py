# src/easy_3dgs/pipeline/feature_extraction/hloc_implementation.py
import logging
from pathlib import Path
from hloc import extract_features
from .base import AbstractFeatureExtractor

class HlocFeatureExtractor(AbstractFeatureExtractor):
    """Concrete implementation for local feature extraction using HLOC."""
    def __init__(self, config: dict):
        self.config = config

    def run(self, image_dir: Path, output_dir: Path):
        logging.info("Step 3/6: Extracting local features...")
        return extract_features.main(self.config, image_dir, output_dir)
