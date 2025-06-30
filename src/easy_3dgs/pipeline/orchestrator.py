# src/easy_3dgs/pipeline/orchestrator.py
import logging
import shutil
from pathlib import Path

# Import concrete implementations from the new modular structure
from .feature_retrieval import HlocFeatureRetriever
from .pair_generation import HlocPairGenerator
from .feature_extraction import HlocFeatureExtractor
from .feature_matching import HlocFeatureMatcher
from .reconstruction import HlocReconstructor
from .image_undistortion import PycolmapImageUndistorter

class ReconstructionPipeline:
    """Orchestrates the entire 3D reconstruction pipeline."""
    def __init__(
        self,
        retrieval_conf: dict,
        feature_conf: dict,
        matcher_conf: dict,
        num_matched_pairs: int = 5,
        mapper_options: dict = None
    ):
        self.retrieval_conf = retrieval_conf
        self.feature_conf = feature_conf
        self.matcher_conf = matcher_conf
        self.num_matched_pairs = num_matched_pairs
        self.mapper_options = mapper_options or {"ba_global_function_tolerance": 0.000001}

        # Instantiate steps with their configurations
        self.retriever = HlocFeatureRetriever(self.retrieval_conf)
        self.pair_generator = HlocPairGenerator()
        self.extractor = HlocFeatureExtractor(self.feature_conf)
        self.matcher = HlocFeatureMatcher(self.matcher_conf)
        self.reconstructor = HlocReconstructor()
        self.undistorter = PycolmapImageUndistorter()

    def run(self, image_dir: Path, output_dir: Path, clean_output: bool = True):
        """
        Executes the full reconstruction pipeline.

        Args:
            image_dir (Path): The directory containing the input images.
            output_dir (Path): The directory where results will be saved.
            clean_output (bool): If True, cleans the output directory before starting.
        """
        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")

        if clean_output:
            logging.info(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        sfm_pairs_path = output_dir / f"pairs-{self.retrieval_conf['output']}.txt"
        sfm_dir = output_dir / f"sfm_{self.feature_conf['output']}+{self.matcher_conf['output']}"

        # --- Execute steps in order ---
        retrieval_path = self.retriever.run(image_dir, output_dir)
        self.pair_generator.run(retrieval_path, sfm_pairs_path, self.num_matched_pairs)
        feature_path = self.extractor.run(image_dir, output_dir)
        match_path = self.matcher.run(sfm_pairs_path, self.feature_conf["output"], output_dir)
        self.reconstructor.run(sfm_dir, image_dir, sfm_pairs_path, feature_path, match_path, self.mapper_options)
        self.undistorter.run(sfm_dir, image_dir)

        logging.info(f"\nPipeline finished. Results in: {sfm_dir}")
        return sfm_dir
