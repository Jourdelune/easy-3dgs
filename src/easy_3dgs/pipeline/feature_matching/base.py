# src/easy_3dgs/pipeline/feature_matching/base.py
from abc import ABC, abstractmethod
from pathlib import Path

class AbstractFeatureMatcher(ABC):
    """Abstract class to match features between image pairs."""
    @abstractmethod
    def run(self, pairs_path: Path, feature_output_name: str, output_dir: Path):
        pass
