import logging

logger = logging.getLogger("pixachu")


def get_logger() -> logging.Logger:
    """
    Returns the logger used by the pixachu package.
    """
    return logger


def set_verbosity(level: int | str):
    """
    Set the logging level for the pixachu logger.

    Args:
        level (int or str): Logging level (e.g., logging.INFO or "INFO").
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)


def add_stream_handler(level: int = logging.INFO, fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"):
    """
    Add a stream handler to the pixachu logger if not already present.

    Args:
        level (int): Logging level for the handler.
        fmt (str): Format string for log messages.
    """
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)


add_stream_handler()
