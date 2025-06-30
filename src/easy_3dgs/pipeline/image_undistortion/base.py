# src/easy_3dgs/pipeline/image_undistortion/base.py
from abc import ABC, abstractmethod
from pathlib import Path

class AbstractImageUndistorter(ABC):
    """Abstract class to undistort images."""
    @abstractmethod
    def run(self, sfm_dir: Path, image_dir: Path):
        pass
