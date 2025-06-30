# src/easy_3dgs/pipeline/pair_generation/base.py
from abc import ABC, abstractmethod
from pathlib import Path

class AbstractPairGenerator(ABC):
    """Abstract class to generate image pairs from retrieval results."""
    @abstractmethod
    def run(self, retrieval_path: Path, output_path: Path, num_matched: int):
        pass
