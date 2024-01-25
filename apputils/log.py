import logging

from config.appconfig import LOG_FILE, SEVERITY

logger = logging.Logger(name='Applogger')
file_handler = logging.FileHandler(LOG_FILE)
logger.addHandler(file_handler)

async def write_log(severity,message):
    if(severity==SEVERITY.INFO):
        logger.info(msg=message)
        return
    if(severity==SEVERITY.ERROR):
        logger.error(msg=message)
        return
    if(severity==SEVERITY.DEBUG):
        logger.debug(msg=message)
        return