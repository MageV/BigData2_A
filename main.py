import asyncio
import datetime as dt
import glob
import os
from concurrent.futures import as_completed

from apputils.archivers import ArchiveManager
from apputils.log import write_log
from apputils.observers import ZipFileObserver
from apputils.storages import ResultStorage
from config.appconfig import *
from parsers.webparser import WebScraper
from parsers.xmlparser import XmlReadSingle

if __name__ == '__main__':

    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    filelist: list = glob.glob(ZIP_FOIV + '*.zip')
    storage=ResultStorage()
    archive_manager = ArchiveManager()
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        for _ in filelist:
            os.remove(_)
        observer = ZipFileObserver()
        parser = WebScraper()
        parser.get()
        asyncio.run(archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE))
    elif APP_FILE_DEBUG:
        asyncio.run(archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE))
    elif XML_FILE_DEBUG:
        xml_importer = XmlReadSingle()
        filelist = glob.glob(XML_STORE + '*.xml')
        futures=[pool.submit(xml_importer.loadxml(_)) for (_) in filelist]
        for future in as_completed(futures):
            storage.append(future.result())
        asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
