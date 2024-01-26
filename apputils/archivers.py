import asyncio
import glob
from zipfile import ZipFile

from aiofiles import os

from abstract.meta import MetaSingleton
from apputils.log import write_log
from config.appconfig import *


class ArchiveManager(metaclass=MetaSingleton):
    def __init__(self,archiver=DEFAULT_ARC):
        self.__arc = archiver

    async def extract(self, source, dest,is_delete_source=True):
        if self.__arc == ARC_TYPES.ZIP:
            with ZipFile(source, 'r') as handle:
                handle.extractall(dest)
                if is_delete_source:
                    await os.remove(source)
            await write_log(message=f'file unpacked:{source}', severity=SEVERITY.INFO)
