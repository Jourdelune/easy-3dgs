# easy-3dgs

A Python library for 3D Gaussian Splatting, designed to be easy to use and extend.

## Features

- **Modular Pipeline:** The reconstruction process is broken down into a series of modular steps, including feature retrieval, pair generation, feature extraction, feature matching, reconstruction, and image undistortion.
- **Extensible:** Each step in the pipeline is designed to be easily replaceable with custom implementations.
- **HLOC-Based:** Leverages the power of Hierarchical Localization (HLOC) for robust and accurate 3D reconstruction.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### Installation

#### Install from PyPI
You can install the library directly from PyPI using pip:

```bash
pip install git+https://github.com/Jourdelune/easy-3dgs.git
pip install git+https://github.com/cvg/Hierarchical-Localization.git
```

#### Install from Source

1. **Clone the repository:**

   ```bash
   git clone --recursive https://github.com/Jourdelune/easy-3dgs.git
   cd easy-3dgs
   ```

2. **Install dependencies:**

   ```bash
   pip install -e .
   pip install git+https://github.com/cvg/Hierarchical-Localization.git
   ```

### Usage

The main entry point for the library is the `ReconstructionPipeline` class. Here's a simple example of how to use it:

```python
from pathlib import Path
from easy_3dgs.pipeline import ReconstructionPipeline
from hloc import extract_features, match_features

# Define paths
image_directory = Path("path/to/your/images")
output_directory = Path("outputs/reconstruction")

# Define configurations
retrieval_config = extract_features.confs["netvlad"]
feature_config = extract_features.confs["superpoint_aachen"]
matcher_config = match_features.confs["superpoint+lightglue"]

# Create and run the pipeline
pipeline = ReconstructionPipeline(
    retrieval_conf=retrieval_config,
    feature_conf=feature_config,
    matcher_conf=matcher_config,
    num_matched_pairs=5,
)
pipeline.run(image_directory, output_directory)
```

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
│       │   ├── orchestrator.py
│       │   ├── feature_extraction
│       │   ├── feature_matching
│       │   ├── feature_retrieval
│       │   ├── image_undistortion
│       │   ├── pair_generation
│       │   └── reconstruction
│       └── third_party
│           └── Hierarchical-Localization
└── test.py
```

- **`run_pipeline.py`**: An example script demonstrating how to use the `ReconstructionPipeline`.
- **`src/easy_3dgs/pipeline`**: Contains the core logic for the reconstruction pipeline.
- **`src/easy_3dgs/pipeline/orchestrator.py`**: The main entry point for the library, containing the `ReconstructionPipeline` class.
- **`src/easy_3dgs/third_party`**: Contains third-party libraries, such as Hierarchical Localization.

## To Do 

- [ ] Add 3d Gaussian Splatting

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
