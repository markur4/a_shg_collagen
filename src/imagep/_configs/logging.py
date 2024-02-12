"""Configuration of logging"""

# %%
import logging


# %%
def configure_logger():
    logger = logging.getLogger()

    ### Add handler to only display warnings in the console
    warningstream = logging.StreamHandler()
    warningstream.setLevel(logging.WARNING)

    ### Add handler to log all messages to FILE
    logfile_complete = logging.FileHandler("log_complete.log")
    # > Level DEBUG includes INFO, WARNING, ERROR, and CRITICAL 
    logfile_complete.setLevel(logging.DEBUG)  

    ### Add handler to log image processing to FILE
    logfile_history = logging.FileHandler("log_processing_history.log")
    # > Level INFO includes WARNING, ERROR and CRITICAL
    logfile_history.setLevel(logging.INFO)
    
    ### Add handlers
    logger.addHandler(warningstream)
    logger.addHandler(logfile_complete)
    logger.addHandler(logfile_history)