# from src.models.basic_logger import aLog

import logging
from pathlib import Path
from datetime import datetime


class BasicLogger:
    singleton: logging.Logger | None = None

    @staticmethod
    def debug(message: str, *args, **kwargs) -> None:
        BasicLogger.log().debug(message, *args, **kwargs)

    @staticmethod
    def info(message: str, *args, **kwargs) -> None:
        BasicLogger.log().info(message, *args, **kwargs)

    @staticmethod
    def warning(message: str, *args, **kwargs) -> None:
        BasicLogger.log().warning(message, *args, **kwargs)

    @staticmethod
    def error(message: str, *args, **kwargs) -> None:
        BasicLogger.log().error(message, *args, **kwargs)

    @staticmethod
    def critical(message: str, *args, **kwargs) -> None:
        BasicLogger.log().critical(message, *args, **kwargs)

    @classmethod
    def init(cls, root_path: Path):
        if cls.singleton is None:
            cls.singleton = logging.getLogger("basic_logger")
            cls.singleton.setLevel(logging.DEBUG)

            file_handler = logging.FileHandler(root_path / ('log_' + datetime.now().strftime("%Y-%m-%d_%H-%M") + '.log'), encoding="utf-8")
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(formatter)
            cls.singleton.addHandler(file_handler)

            cls.singleton.propagate = False

    @classmethod
    def log(cls) -> logging.Logger:
        return cls.singleton

aLog = BasicLogger
