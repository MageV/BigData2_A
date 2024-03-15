import asyncio
import os
from zipfile import ZipFile

from abstract.abstractions import MetaSingleton
from apputils.log import write_log
from config.appconfig import *


class ArchiveManager(metaclass=MetaSingleton):
    def __init__(self, archiver=DEFAULT_ARC):
        self.__arc = archiver

    def extract(self, source, dest, is_delete_source=False):
        if self.__arc == ARC_TYPES.ZIP:
            with ZipFile(source, 'r') as handle:
                handle.extractall(dest)
                if is_delete_source:
                    os.remove(source)
            asyncio.run(write_log(message=f'finished unpack:{source}', severity=SEVERITY.INFO))


