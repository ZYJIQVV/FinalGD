# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.2"

from gd.data.explorer.explorer import Explorer
from gd.models import RTDETR, SAM, YOLO, YOLOWorld
from gd.models.fastsam import FastSAM
from gd.models.nas import NAS
from gd.utils import ASSETS, SETTINGS
from gd.utils.checks import check_yolo as checks
from gd.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
