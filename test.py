import sys
import os

import sys

from easy_3dgs.pipeline import ReconstructionPipeline

from SuperGluePretrainedNetwork.models.superglue import SuperGlue as SG  # noqa: E402

print("Import successful!")  # Si vous atteignez cette ligne, l'importation a réussi

import shutil
import time
from pathlib import Path

import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction


images = Path("/home/jourdelune/dev/colmap-api/takeout-1-001")

shutil.rmtree("outputs/sfm", ignore_errors=True)
outputs = Path("outputs/sfm/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+lightglue"

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superpoint+lightglue"]
# matcher_conf = match_dense.confs["loftr_aachen"]

time1 = time.time()

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)

match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)
model = reconstruction.main(
    sfm_dir,
    images,
    sfm_pairs,
    feature_path,
    match_path,
    mapper_options={"ba_global_function_tolerance": 0.000001},
)

pycolmap.undistort_images(output_path=sfm_dir, image_path=images, input_path=sfm_dir),
