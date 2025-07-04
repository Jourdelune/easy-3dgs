# src/easy_3dgs/pipeline/orchestrator.py
import logging
import shutil
from pathlib import Path
from typing import Optional, Type

from .feature_extraction import HlocFeatureExtractor
from .feature_extraction.base import AbstractFeatureExtractor
from .feature_matching import HlocFeatureMatcher
from .feature_matching.base import AbstractFeatureMatcher
from .feature_retrieval import HlocFeatureRetriever
from .feature_retrieval.base import AbstractFeatureRetriever
from .image_undistortion import PycolmapImageUndistorter
from .image_undistortion.base import AbstractImageUndistorter
from .pair_generation import HlocPairGenerator
from .pair_generation.base import AbstractPairGenerator
from .reconstruction import HlocReconstructor
from .reconstruction.base import AbstractReconstructor
from .resizer_image import ImageMagickResizer
from .resizer_image.base import BaseResizer


class ReconstructionPipeline:
    """Orchestrates a flexible 3D reconstruction pipeline."""

    def __init__(
        self,
        retriever_class: Optional[
            Type[AbstractFeatureRetriever]
        ] = HlocFeatureRetriever,
        pair_generator_class: Optional[Type[AbstractPairGenerator]] = HlocPairGenerator,
        extractor_class: Optional[
            Type[AbstractFeatureExtractor]
        ] = HlocFeatureExtractor,
        matcher_class: Optional[Type[AbstractFeatureMatcher]] = HlocFeatureMatcher,
        reconstructor_class: Optional[Type[AbstractReconstructor]] = HlocReconstructor,
        undistorter_class: Optional[
            Type[AbstractImageUndistorter]
        ] = PycolmapImageUndistorter,
        resizer_class: Optional[Type[BaseResizer]] = ImageMagickResizer,
        retrieval_conf: Optional[dict] = None,
        feature_conf: Optional[dict] = None,
        matcher_conf: Optional[dict] = None,
        num_matched_pairs: int = 5,
        mapper_options: Optional[dict] = None,
    ):
        # --- Store configurations ---
        self.retrieval_conf = retrieval_conf
        self.feature_conf = feature_conf
        self.matcher_conf = matcher_conf
        self.num_matched_pairs = num_matched_pairs
        self.mapper_options = mapper_options or {}

        # --- Conditionally instantiate steps ---
        self.retriever = (
            retriever_class(self.retrieval_conf)
            if retriever_class and self.retrieval_conf
            else None
        )
        self.pair_generator = pair_generator_class() if pair_generator_class else None
        self.extractor = (
            extractor_class(self.feature_conf)
            if extractor_class and self.feature_conf
            else None
        )
        self.matcher = (
            matcher_class(self.matcher_conf)
            if matcher_class and self.matcher_conf
            else None
        )
        self.reconstructor = reconstructor_class() if reconstructor_class else None
        self.undistorter = undistorter_class() if undistorter_class else None
        self.resizer = resizer_class() if resizer_class else None

    def run(
        self,
        image_dir: Path,
        output_dir: Path,
        clean_output: bool = True,
        resize: bool = False,
        retrieval_path: Optional[Path] = None,
        sfm_pairs_path: Optional[Path] = None,
        feature_path: Optional[Path] = None,
        match_path: Optional[Path] = None,
        sfm_dir: Optional[Path] = None,
    ):
        """
        Executes the reconstruction pipeline based on the configured steps.

        Args:
            image_dir (Path): The directory containing the input images.
            output_dir (Path): The directory where results will be saved.
            clean_output (bool): If True, cleans the output directory before starting.
            resize (bool): If True, resizes the images after undistortion.
            retrieval_path (Path, optional): Path to pre-computed retrieval results.
            sfm_pairs_path (Path, optional): Path to pre-computed pair lists.
            feature_path (Path, optional): Path to pre-computed local features.
            match_path (Path, optional): Path to pre-computed feature matches.
            sfm_dir (Path, optional): Path to the main SfM directory.
        """
        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")

        if clean_output:
            logging.info(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Define default output paths if not provided ---
        if self.retriever and not retrieval_path:
            retrieval_path = output_dir / f"pairs-{self.retrieval_conf['output']}.txt"
        if self.pair_generator and not sfm_pairs_path:
            sfm_pairs_path = output_dir / f"pairs-{self.retrieval_conf['output']}.txt"
        if self.reconstructor and not sfm_dir:
            sfm_dir = output_dir / f"sfm"

        # --- Execute steps in order ---
        if self.retriever:
            retrieval_path = self.retriever.run(image_dir, output_dir)
        elif not retrieval_path:
            raise ValueError(
                "Retriever step is skipped, but 'retrieval_path' was not provided."
            )

        if self.pair_generator:
            self.pair_generator.run(
                retrieval_path, sfm_pairs_path, self.num_matched_pairs
            )
        elif not sfm_pairs_path:
            raise ValueError(
                "Pair generator step is skipped, but 'sfm_pairs_path' was not provided."
            )

        if self.extractor:
            feature_path = self.extractor.run(image_dir, output_dir)
        elif not feature_path:
            raise ValueError(
                "Extractor step is skipped, but 'feature_path' was not provided."
            )

        if self.matcher:
            match_path = self.matcher.run(
                sfm_pairs_path, self.feature_conf["output"], output_dir
            )
        elif not match_path:
            raise ValueError(
                "Matcher step is skipped, but 'match_path' was not provided."
            )

        if self.reconstructor:
            self.reconstructor.run(
                sfm_dir,
                image_dir,
                sfm_pairs_path,
                feature_path,
                match_path,
                self.mapper_options,
            )
        elif not sfm_dir:
            raise ValueError(
                "Reconstructor step is skipped, but 'sfm_dir' was not provided."
            )

        if self.undistorter:
            self.undistorter.run(sfm_dir, image_dir)

        if resize and self.resizer:
            self.resizer.main(sfm_dir, [2, 4, 8])

        logging.info(f"\nPipeline finished. Results in: {sfm_dir}")
        return sfm_dir
