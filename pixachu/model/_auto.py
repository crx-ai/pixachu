from transformers import AutoConfig, AutoModel


class AutoRegisterConfigMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AutoConfig.register(cls.model_type, cls, exist_ok=True)


class AutoRegisterModelMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AutoModel.register(cls.config_class, cls, exist_ok=True)
