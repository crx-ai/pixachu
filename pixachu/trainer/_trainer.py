from typing import Any

from transformers import Trainer

from .._model import PixachuModel


class PixachuTrainer(Trainer):
    def compute_loss(
        self,
        model: PixachuModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ):
        pass
