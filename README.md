# easy-3dgs

A Python library for 3D Gaussian Splatting, designed to be easy to use and extend.

## Features

- **Modular Pipeline:** The reconstruction process is broken down into a series of modular steps, including feature retrieval, pair generation, feature extraction, feature matching, reconstruction, and image undistortion.
- **Extensible:** Each step in the pipeline is designed to be easily replaceable with custom implementations.
- **HLOC-Based:** Leverages the power of Hierarchical Localization (HLOC) for robust and accurate 3D reconstruction.
- **Gaussian Splatting:** Includes a pipeline for training and visualizing 3D Gaussian Splatting models.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### Installation

#### Install from Source
You can install the library directly using pip:

```bash
pip install git+https://github.com/Jourdelune/easy-3dgs.git
```

The project can't be installed from PyPI because it requires dependencies that are not available on PyPI.

#### Install for Development

1. **Clone the repository:**

   ```bash
   git clone --recursive https://github.com/Jourdelune/easy-3dgs.git
   cd easy-3dgs
   ```

2. **Install dependencies:**

   ```bash
   pip install -e .
   ```

### Usage

The main entry point for the library is the `ReconstructionPipeline` and `GaussianSplattingPipeline` classes, which orchestrate the entire reconstruction and Gaussian Splatting process.
Expect the first run to take some time because `gsplat` will be compiled; subsequent runs will be faster.

```python
import logging
from pathlib import Path

from hloc import extract_features, match_dense

from easy_3dgs.pipeline import GaussianSplattingPipeline, ReconstructionPipeline
from easy_3dgs.pipeline.feature_matching.hloc_implementation import (
    HlocDenseFeatureMatcher,
)
from easy_3dgs.pipeline.resizer_image.pillow_implementation import PillowResizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


image_directory = Path("PATH_TO_YOUR_IMAGES")  # Replace with your image directory
output_directory = Path("outputs/reconstruction")

retrieval_config = extract_features.confs["netvlad"]
matcher_config = match_dense.confs["loftr_superpoint"]

reconstruction_pipeline = ReconstructionPipeline(
    resizer_class=PillowResizer,
    extractor_class=None,  # No feature extraction step
    matcher_class=HlocDenseFeatureMatcher,
    retrieval_conf=retrieval_config,
    matcher_conf=matcher_config,
    num_matched_pairs=5,
    mapper_options={"ba_global_function_tolerance": 0.000001},
)

reconstruction_pipeline.run(image_directory, output_directory, resize=True)

gaussian_splatting_pipeline = GaussianSplattingPipeline(
    data_factor=4,
    result_dir="./results/",
    strategy_type="mcmc",  # or "default" for default strategy, there are much more options available (see the implementation of the class for more details)
)

gaussian_splatting_results_dir = gaussian_splatting_pipeline.train(
    output_directory / "sfm"
)

logging.info(f"Gaussian Splatting results in: {gaussian_splatting_results_dir}")
```

There are also example scripts in the `examples` directory that demonstrate how to use the library for different tasks, such as dense reconstruction.

## Project Structure

The project is organized as follows:

```
├── pyproject.toml
├── README.md
├── run_pipeline.py
├── src
│   └── easy_3dgs
│       ├── __init__.py
│       ├── pipeline
│       │   ├── __init__.py
│       │   ├── gaussian_splatting_pipeline.py
│       │   ├── sfm_pipeline.py
│       │   ├── feature_extraction
│       │   ├── feature_matching
│       │   ├── feature_retrieval
│       │   ├── gaussian_splatting
│       │   ├── image_undistortion
│       │   ├── pair_generation
│       │   └── reconstruction
│       └── third_party
│           └── Hierarchical-Localization
```

- **`run_pipeline.py`**: An example script demonstrating how to use the `ReconstructionPipeline` and `GaussianSplattingPipeline`.
- **`src/easy_3dgs/pipeline`**: Contains the core logic for the reconstruction and splatting pipelines.
- **`src/easy_3dgs/pipeline/sfm_pipeline.py`**: The main entry point for the reconstruction pipeline.
- **`src/easy_3dgs/pipeline/gaussian_splatting_pipeline.py`**: The main entry point for the Gaussian Splatting pipeline.
- **`src/easy_3dgs/third_party`**: Contains third-party libraries, such as Hierarchical Localization.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.