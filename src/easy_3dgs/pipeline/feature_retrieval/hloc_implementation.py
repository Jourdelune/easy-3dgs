# src/easy_3dgs/pipeline/feature_retrieval/hloc_implementation.py
import logging
from pathlib import Path
from hloc import extract_features
from .base import AbstractFeatureRetriever

class HlocFeatureRetriever(AbstractFeatureRetriever):
    """Concrete implementation for feature retrieval using HLOC."""
    def __init__(self, config: dict):
        self.config = config

    def run(self, image_dir: Path, output_dir: Path):
        logging.info("Step 1/6: Extracting features for retrieval...")
        return extract_features.main(self.config, image_dir, output_dir)
