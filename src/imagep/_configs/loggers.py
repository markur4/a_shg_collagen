"""Configuration of logging

Used this awesome tutorial to set up logging:
https://www.youtube.com/watch?v=9L77QExPmI0
"""

# %%
# from typing import override
import os
from collections import defaultdict

import logging

from pathlib import Path


# %%
# == Filters ===========================================================
class NonErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno < logging.ERROR


class InfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno == logging.INFO

#%%
# == Utils =============================================================
def wrap_lines(s: str, width: int = 40) -> str:
    """Wrap lines to a maximum width"""
    return "\n".join(
        [s[i : i + width] for i in range(0, len(s), width)]
    )
    
if __name__ == "__main__":
    print(wrap_lines("This is a very long string that needs to be wrapped to fit in the console window."))
# %%
# == Locations =========================================================
LOGDIR = Path("_imagep_logs")
LOGDIR.mkdir(exist_ok=True)
file = (Path(LOGDIR, "debug.log"),)

# %%
# == Configure Handlers ================================================

# !! Handlers are Configured on root logger
LOGGING_CONFIG = dict(
    version=1,  # > Required
    disable_existing_loggers=False,  # > Allows messages from other libraries
    formatters={
        "short": {
            "()": logging.Formatter,
            "format": (
                "%(levelname)s"
                " [%(module)s|%(funcName)s]"
                # " (l:%(lineno)d):"
                ": %(message)s"
            ),
        },
        "long": {
            "()": logging.Formatter,
            "format": (
                "[ %(name)-13s: %(levelname)-8s %(asctime)44s ]"
                "\n[ %(module)-27s | %(funcName)-29s | l:%(lineno)4d ]"
                + "\n%(message)s\n"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S %Z",
        },
        "history": {
            "()": logging.Formatter,
            "format": (
                "[ %(asctime)-68s ]"
                + "\n[ %(module)-27s | %(funcName)-29s ]"
                + wrap_lines("\n%(message)s\n")
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S %Z",
        },
    },
    handlers={
        # > Level DEBUG includes INFO, WARNING, ERROR, and CRITICAL
        # > Level INFO includes WARNING, ERROR and CRITICAL
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "short",
            "stream": "ext://sys.stdout",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            # "level": "WARNING",  # !! Causes duplicate messages
            "formatter": "short",
            "stream": "ext://sys.stderr",
        },
        "debugfile": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "long",
            "filename": Path(LOGDIR, "debug.log"),
            "maxBytes": 2 * 1e6,  # > 2 * 1e6 = 2 MegaByte
            "backupCount": 3,
        },
        # "debugfile_from_history": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "DEBUG",
        #     "formatter": "history",  # > Different formatter, same file
        #     "filename": Path(LOGDIR, "debug.log"),
        #     "maxBytes": 2 * 1e6,  # > 2 * 1e6 = 2 MegaByte
        #     "backupCount": 3,
        # },
        "historyfile": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "history",
            "filename": Path(LOGDIR, "history.log"),
        },
    },
    filters={
        "nonerror": {
            "()": NonErrorFilter,
        },
        "info": {
            "()": InfoFilter,
        },
    },
    loggers={
        "debuglogger": {
            "level": "DEBUG",
            "handlers": [
                "stdout", 
                "stderr",
                "debugfile",
            ],
        },
        "historylogger": {
            "level": "INFO",  # > Includes warnings, filter them out
            "handlers": [
                # "debugfile_from_history",  # > To log everything
                "historyfile",
                "debugfile"
            ],
            "filters": ["info"],  # > Only Info
        },
    },
)


# == Make loggers ======================================================
#!! Handlers are Configured on root logger
DEBUG_LOGGER = logging.getLogger("debuglogger")
HISTORY_LOGGER = logging.getLogger("historylogger")


# == Configure at Root Logger ==========================================
logging.config.dictConfig(LOGGING_CONFIG)


# %%
def main():

    # logging.config.dictConfig(LOGGING_CONFIG)
    # ALL_LOGGER = logging.getLogger("all_logger")
    ### Debug
    DEBUG_LOGGER.debug("This is a debug message")
    DEBUG_LOGGER.info("This is an info message")
    DEBUG_LOGGER.warning("This is a warning message")
    DEBUG_LOGGER.error("This is an error message")
    DEBUG_LOGGER.critical("This is a critical message")
    try:
        1 / 0
    except Exception as e:
        DEBUG_LOGGER.exception("An exception occurred")

    ### History of image processing
    HISTORY_LOGGER.info("This is an HISTORY info message")
    HISTORY_LOGGER.info("This is an long HISTORY info message, This is an long HISTORY info message, This is an long HISTORY info message, This is an long HISTORY info message, This is an long HISTORY info message")
    HISTORY_LOGGER.warning(
        "This is a HISTORY warning message"
    )  # > this shouldn't be logged (?)


if __name__ == "__main__":
    main()

# %%
