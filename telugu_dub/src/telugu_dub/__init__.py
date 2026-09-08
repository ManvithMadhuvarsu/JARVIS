"""English -> Telugu video dubbing with lip synchronisation."""
from .config import Config
from .pipeline import Pipeline
from .schema import Manifest, Segment

__all__ = ["Config", "Pipeline", "Manifest", "Segment"]
__version__ = "0.1.0"
