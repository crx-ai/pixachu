from typing import TypedDict, Unpack

from transformers import Dinov2Config

from ._auto import AutoRegisterConfigMixin


class PixachuConfigDict(TypedDict, total=False):
    # These from Dinov2Config
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    mlp_ratio: int
    hidden_act: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    initializer_range: float
    layer_norm_eps: float
    image_size: int
    patch_size: int
    num_channels: int
    qkv_bias: bool
    layerscale_value: float
    drop_path_rate: float
    use_swiglu_ffn: bool
    out_features: list[str] | None
    out_indices: list[int] | None
    apply_layernorm: bool
    reshape_hidden_states: bool
    use_mask_token: bool

    # These are specific to Pixachu
    character_pixel_size: int


class PixachuConfig(AutoRegisterConfigMixin, Dinov2Config):
    model_type = "pixachu"

    def __init__(self, **kwargs: Unpack[PixachuConfigDict]):
        super().__init__(**kwargs)

        # Pixachu-specific parameters
        self.character_pixel_size = kwargs.get("character_pixel_size", 24)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_for_auto_class()
