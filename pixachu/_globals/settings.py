import pydantic_settings


class PixachuSettings(pydantic_settings.BaseSettings):
    log_level: int | str = "INFO"

    class Config:
        env_prefix = "PIXACHU_"
        case_sensitive = False
