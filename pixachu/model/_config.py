from typing import Literal, TypedDict, Unpack

from transformers import ModernBertConfig


class PixachuConfigDict(TypedDict, total=False):
    # These from ModernBertConfig
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    hidden_activation: str
    max_position_embeddings: int
    initializer_range: float
    initializer_cutoff_factor: float
    norm_eps: float
    norm_bias: bool
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    cls_token_id: int
    sep_token_id: int
    global_rope_theta: float
    attention_bias: bool
    attention_dropout: float
    global_attn_every_n_layers: int
    local_attention: int
    local_rope_theta: float
    embedding_dropout: float
    mlp_bias: bool
    mlp_dropout: float
    decoder_bias: bool
    classifier_pooling: Literal["cls", "mean"]
    classifier_dropout: float
    classifier_bias: bool
    classifier_activation: str
    deterministic_flash_attn: bool
    sparse_prediction: bool
    sparse_pred_ignore_index: int
    reference_compile: bool
    repad_logits_with_grad: bool

    # These are specific to Pixachu
    patch_pixel_size: int
    logit_scale: float
    logit_bias: float


class PixachuConfig(ModernBertConfig):
    model_type = "pixachu"

    def __init__(self, **kwargs: Unpack[PixachuConfigDict]):
        super().__init__(**kwargs)

        # Pixachu-specific parameters
        self.patch_pixel_size = kwargs.get("patch_pixel_size", 14)
        self.logit_scale = kwargs.get("logit_scale", 2.6592)
        self.logit_bias = kwargs.get("logit_bias", 0.0)
