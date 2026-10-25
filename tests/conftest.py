import sys
from pathlib import Path


# Allow running the source checkout without installing the training extras.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
