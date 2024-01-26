import asyncio
import datetime as dt
import glob
import os

from apputils.archivers import ArchiveManager
from apputils.log import write_log
from apputils.observers import ZipFileObserver
from config.appconfig import *
from parsers.webparser import WebScraper
from parsers.xmlparser import XmlReadSingle

if __name__ == '__main__':

    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    filelist: list = glob.glob(ZIP_FOIV + '*.zip')
    archive_manager = ArchiveManager()
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        for _ in filelist:
            os.remove(_)
        observer = ZipFileObserver()
        parser = WebScraper()
        parser.get()
    elif APP_FILE_DEBUG:
        asyncio.run(archive_manager.extract(is_delete=False, source=APP_FILE_DEBUG_NAME, dest=XML_STORE))
    elif XML_FILE_DEBUG:
        xml_importer = XmlReadSingle()
        filelist = glob.glob(XML_STORE + '*.xml')
        for (_) in filelist:
            xml_importer.loadxml(_)

    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
