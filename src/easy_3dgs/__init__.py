import sys
from pathlib import Path

# Add third-party dependencies to the Python path
sys.path.insert(
    0,
    str(Path(__file__).parent / "third_party/Hierarchical-Localization/third_party/"),
)