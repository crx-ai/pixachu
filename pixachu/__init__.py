from importlib.metadata import version

from ._globals import PixachuSettings, get_logger, set_verbosity
from ._model import PixachuConfig, PixachuForMaskedImageModeling, PixachuModel, PixachuProcessor

__version__ = version("pixachu")

__all__ = [
    "__version__",
    "get_logger",
    "set_verbosity",
    "PixachuSettings",
    "PixachuModel",
    "PixachuConfig",
    "PixachuForMaskedImageModeling",
    "PixachuProcessor",
]
