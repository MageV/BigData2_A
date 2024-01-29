import asyncio
import datetime as dt
import glob
import os
from multiprocessing import Pool, freeze_support

from apputils.archivers import ArchiveManager
from apputils.log import write_log
from apputils.observers import ZipFileObserver
from apputils.storages import ResultStorage
from config.appconfig import *
from parsers.webparser import WebScraper
from parsers.xmlparser import loadxml

storage = ResultStorage()

if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    filelist: list = glob.glob(ZIP_FOIV + '*.zip')
    archive_manager = ArchiveManager()
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        for _ in filelist:
            os.remove(_)
        observer = ZipFileObserver()
        parser = WebScraper()
        store = parser.get()
        asyncio.run(archive_manager.extract(source=store, dest=XML_STORE))
    elif APP_FILE_DEBUG:
        asyncio.run(archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE))
    elif XML_FILE_DEBUG:
        filelist = glob.glob(XML_STORE + '*.xml')
        processors=multiprocessing.cpu_count() - 2
        with Pool(processors)as pool:
            for result in pool.map(loadxml, filelist, chunksize=processors*2):
                storage.append(result)
        asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
