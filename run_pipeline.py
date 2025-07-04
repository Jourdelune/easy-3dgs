# run_pipeline.py
import logging
from pathlib import Path

from hloc import extract_features, match_features

from easy_3dgs.pipeline import ReconstructionPipeline
from easy_3dgs.pipeline.reconstruction import HlocReconstructor
from easy_3dgs.pipeline.image_undistortion import PycolmapImageUndistorter
from easy_3dgs.pipeline.resizer_image.pillow_implementation import PillowResizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


image_directory = Path("/home/jourdelune/Images/colmap/images/")
output_directory = Path("outputs/reconstruction")

retrieval_config = extract_features.confs["netvlad"]
feature_config = extract_features.confs["superpoint_aachen"]
matcher_config = match_features.confs["superpoint+lightglue"]

# You can now customize the pipeline by passing different classes or None to skip steps.
# Example: To skip feature retrieval, set `retriever_class=None` and provide a `retrieval_path` to the `run` method.
pipeline = ReconstructionPipeline(
    resizer_class=PillowResizer,
    retrieval_conf=retrieval_config,
    feature_conf=feature_config,
    matcher_conf=matcher_config,
    num_matched_pairs=5,
    mapper_options={"ba_global_function_tolerance": 0.000001},
)

# Run the full pipeline
pipeline.run(image_directory, output_directory, resize=True)
