# Pipeline for sparse reconstruction and Gaussian Splatting
# This script demonstrates how to run a sparse reconstruction pipeline using hloc for feature extraction and matching
# and then apply Gaussian Splatting for rendering the reconstructed scene.

import logging
from pathlib import Path

from hloc import extract_features, match_features

from easy_3dgs.pipeline import GaussianSplattingPipeline, ReconstructionPipeline
from easy_3dgs.pipeline.resizer_image.pillow_implementation import PillowResizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


image_directory = Path("PATH_TO_YOUR_IMAGES")  # Replace with your image directory
output_directory = Path("outputs/reconstruction")

retrieval_config = extract_features.confs["netvlad"]
feature_config = extract_features.confs["superpoint_aachen"]
matcher_config = match_features.confs["superpoint+lightglue"]

reconstruction_pipeline = ReconstructionPipeline(
    resizer_class=PillowResizer,
    retrieval_conf=retrieval_config,
    feature_conf=feature_config,
    matcher_conf=matcher_config,
    num_matched_pairs=5,
    mapper_options={"ba_global_function_tolerance": 0.000001},
)

reconstruction_pipeline.run(image_directory, output_directory, resize=True)

gaussian_splatting_pipeline = GaussianSplattingPipeline(
    data_factor=4,
    result_dir="./results/",
    antialiased=True,
    with_ut=True,
    with_eval3d=True,
    strategy_type="mcmc",
)

gaussian_splatting_results_dir = gaussian_splatting_pipeline.train(
    output_directory / "sfm"
)

logging.info(f"Gaussian Splatting results in: {gaussian_splatting_results_dir}")
