import logging
import sys

# Define custom debug levels
DEBUG_L1 = 15  # Between INFO and DEBUG
DEBUG_L2 = 8   # Between DEBUG1 and DEBUG
DEBUG_L3 = 5   # Between DEBUG2 and NOTSET

# Register custom levels
logging.addLevelName(DEBUG_L1, "DEBUG_L1")
logging.addLevelName(DEBUG_L2, "DEBUG_L2")
logging.addLevelName(DEBUG_L3, "DEBUG_L3")

# Add custom logging methods
def debug1(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_L1): 
        self._log(DEBUG_L1, message, args, **kws)

def debug2(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_L2):
        self._log(DEBUG_L2, message, args, **kws)

def debug3(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_L3):
        self._log(DEBUG_L3, message, args, **kws)

# Add methods to Logger class
logging.Logger.debug1 = debug1
logging.Logger.debug2 = debug2
logging.Logger.debug3 = debug3

def setup_logger(name: str, level: int) -> logging.Logger:
    """Setup logger with appropriate configuration."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger