# src/easy_3dgs/pipeline/pair_generation/hloc_implementation.py
import logging
from pathlib import Path
from hloc import pairs_from_retrieval
from .base import AbstractPairGenerator

class HlocPairGenerator(AbstractPairGenerator):
    """Concrete implementation for pair generation using HLOC."""
    def run(self, retrieval_path: Path, output_path: Path, num_matched: int):
        logging.info(f"Step 2/6: Generating {num_matched} image pairs...")
        pairs_from_retrieval.main(retrieval_path, output_path, num_matched=num_matched)
        return output_path
