# run_pipeline.py
import logging
from pathlib import Path

from hloc import extract_features, match_features

from easy_3dgs.pipeline import ReconstructionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


image_directory = Path("/home/jourdelune/dev/colmap-api/takeout-1-001")
output_directory = Path("outputs/sfm_pipeline_refactored")

retrieval_config = extract_features.confs["netvlad"]
feature_config = extract_features.confs["superpoint_aachen"]
matcher_config = match_features.confs["superpoint+lightglue"]

# 3. Create and run the pipeline
pipeline = ReconstructionPipeline(
    retrieval_conf=retrieval_config,
    feature_conf=feature_config,
    matcher_conf=matcher_config,
    num_matched_pairs=5,
)
pipeline.run(image_directory, output_directory)
