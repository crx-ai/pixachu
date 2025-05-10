from typing import TypedDict, Unpack

from transformers import Dinov2Config

from .auto import AutoRegisterConfigMixin


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
    masking_proportion: float
    ema_momentum: float
    student_temp: float
    teacher_temp: float
    koleo_weight: float
    proj_dim: int
    num_prototypes: int


class PixachuConfig(AutoRegisterConfigMixin, Dinov2Config):
    model_type = "pixachu"

    def __init__(self, **kwargs: Unpack[PixachuConfigDict]):
        super().__init__(**kwargs)

        # Pixachu-specific parameters
        self.character_pixel_size = kwargs.get("character_pixel_size", 24)
        self.masking_proportion = kwargs.get("masking_proportion", 0.30)  # 30% is better than 15%
        self.ema_momentum = kwargs.get("ema_momentum", 0.996)
        self.student_temp = kwargs.get("student_temp", 0.1)
        self.teacher_temp = kwargs.get("teacher_temp", 0.04)
        self.koleo_weight = kwargs.get("koleo_weight", 1.0)
        self.proj_dim = kwargs.get("proj_dim", 1024)
        self.num_prototypes = kwargs.get("num_prototypes", 8192)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_for_auto_class()
