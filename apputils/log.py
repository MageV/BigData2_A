import logging


from config.appconfig import *

logger = logging.Logger(name='Applogger')
file_handler = logging.FileHandler(LOG_FILE)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(SEVERITY.INFO.value)
if LOG_TO== 'FILE':
    logger.addHandler(file_handler)
else:
    logger.addHandler(consoleHandler)


async def write_log(severity, message):
    if (severity == SEVERITY.INFO):
        logger.info(msg=message)
        return
    if (severity == SEVERITY.ERROR):
        logger.error(msg=message)
        return
    if (severity == SEVERITY.DEBUG):
        logger.debug(msg=message)
        return
