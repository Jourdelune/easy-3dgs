# src/easy_3dgs/pipeline/feature_retrieval/base.py
from abc import ABC, abstractmethod
from pathlib import Path

class AbstractFeatureRetriever(ABC):
    """Abstract class to extract global features for image retrieval."""
    @abstractmethod
    def run(self, image_dir: Path, output_dir: Path):
        pass
