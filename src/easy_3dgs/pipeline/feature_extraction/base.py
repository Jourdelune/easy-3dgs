# src/easy_3dgs/pipeline/feature_extraction/base.py
from abc import ABC, abstractmethod
from pathlib import Path

class AbstractFeatureExtractor(ABC):
    """Abstract class to extract local features from images."""
    @abstractmethod
    def run(self, image_dir: Path, output_dir: Path):
        pass
