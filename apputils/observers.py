import asyncio
import os
from zipfile import ZipFile

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from apputils.log import logger, write_log
from config.appconfig import *


class _ZipFileHandler(FileSystemEventHandler):
    __filesize=0
    def on_modified(self, event):
        super().on_modified(event)
        filename: str = event.src_path
        if filename.__contains__('zip'):
            __filesize=os.path.getsize(filename)
            if __filesize!=0:
                logger.info(f'file closed:{filename}')
                with ZipFile(filename, 'r') as handle:
                    for item in handle.namelist():
                        handle.extractall(item, XML_STORE)
                        os.remove(filename)
                asyncio.run(write_log(message=f'file unpacked:{filename}', severity=SEVERITY.INFO))


    def on_created(self, event):
        super().on_created(event)
        filename: str = event.src_path
        if filename.__contains__('zip'):
            asyncio.run(write_log(message=f'file created:{filename}',severity=SEVERITY.INFO))

    def on_closed(self, event):
        super().on_closed(event)
        filename: str = event.src_path
        if filename.__contains__('zip'):
            logger.info(f'file closed:{filename}')
            with ZipFile(filename,'r') as handle:
                for item in handle.namelist():
                    handle.extractall(item,XML_STORE)
                    os.remove(filename)
            asyncio.run(write_log(message=f'file unpacked:{filename}',severity=SEVERITY.INFO))



class ZipFileObserver:

    def __init__(self):
        self.__observer = Observer()
        self.__observer.schedule(_ZipFileHandler(), path=ZIP_FOIV, recursive=False)
        self.__observer.start()

    def destroy(self):
        self.__observer.stop()
        self.__observer.join()
