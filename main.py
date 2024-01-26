import asyncio
import glob
import os

from apputils.archivers import ArchiveManager
from apputils.log import write_log
from apputils.observers import ZipFileObserver
from config.appconfig import *
from importers.webparser import WebScraper
import datetime as dt

if __name__ == '__main__':
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    filelist: list = glob.glob(ZIP_FOIV + '*.zip')
    archive_manager = ArchiveManager()
    if not APP_FILE_DEBUG:
        for _ in filelist:
            os.remove(_)
        observer = ZipFileObserver()
        parser = WebScraper()
        parser.get()
    else:
        asyncio.run(archive_manager.extract(is_delete=False, source=APP_FILE_DEBUG_NAME, dest=XML_STORE))

    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
