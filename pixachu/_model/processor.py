from text2pic4ai import BitmapSentenceProcessor
from transformers import AutoProcessor

from .auto import AutoRegisterMixin
from .config import PixachuConfig


class PixachuProcessor(AutoRegisterMixin, BitmapSentenceProcessor, auto_cls=AutoProcessor):
    config_class = PixachuConfig
