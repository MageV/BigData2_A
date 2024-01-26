import asyncio
import os
from zipfile import ZipFile

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from apputils.archivers import ArchiveManager
from apputils.log import logger, write_log
from config.appconfig import *


class _ZipFileHandler(FileSystemEventHandler):

    def on_modified(self, event):
        super().on_modified(event)
        __filesize: int = 0
        filename: str = event.src_path
        if filename.__contains__('_new.zip') and event.is_directory != True:
            try:
                zip_file = os.open(filename, 'r')
                os.close(zip_file)
                logger.info(f'file closed:{filename}')
            except:
                pass

    def on_created(self, event):
        super().on_created(event)
        filename: str = event.src_path
        if filename.__contains__('zip'):
            asyncio.run(write_log(message=f'file created:{filename}', severity=SEVERITY.INFO))

    def on_closed(self, event):
        super().on_closed(event)
        filename: str = event.src_path
        if filename.__contains__('zip'):
            logger.info(f'file closed:{filename}')
            archive_manager=ArchiveManager()
            archive_manager.extract(source=filename,dest=XML_STORE)
            asyncio.run(write_log(message=f'file unpacked:{filename}', severity=SEVERITY.INFO))


class ZipFileObserver:

    def __init__(self):
        self.__observer = Observer()
        self.__observer.schedule(_ZipFileHandler(), path=ZIP_FOIV, recursive=False)
        self.__observer.start()

    def destroy(self):
        self.__observer.stop()
        self.__observer.join()
