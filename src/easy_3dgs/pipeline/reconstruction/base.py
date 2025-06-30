# src/easy_3dgs/pipeline/reconstruction/base.py
from abc import ABC, abstractmethod
from pathlib import Path

class AbstractReconstructor(ABC):
    """Abstract class to perform 3D reconstruction."""
    @abstractmethod
    def run(self, sfm_dir: Path, image_dir: Path, pairs_path: Path, feature_path: Path, match_path: Path, mapper_options: dict = None):
        pass
