from transformers import AutoConfig, AutoModel

from .._globals import get_logger


class AutoRegisterConfigMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AutoConfig.register(cls.model_type, cls, exist_ok=True)


class AutoRegisterMixin:
    def __init_subclass__(cls, **kwargs):
        auto_cls = kwargs.pop("auto_cls", AutoModel)

        if isinstance(auto_cls, AutoConfig):
            logger = get_logger()
            exc = ValueError("`AutoRegisterConfigMixin` should be used if `auto_cls` is `AutoConfig`")
            logger.warning(exc.args[0])

            raise exc

        super().__init_subclass__(**kwargs)
        auto_cls.register(cls.config_class, cls, exist_ok=True)
