"""Resources for a source checkout, independent of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"
