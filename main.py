import asyncio

from apputils.log import write_log
from apputils.observers import ZipFileObserver
from config.appconfig import SEVERITY
from importers.webparser import WebScraper
import datetime as dt

if __name__ == '__main__':
    asyncio.run(write_log(message=f'Started at{dt.datetime.now()}',severity=SEVERITY.INFO))
    observer = ZipFileObserver()
    parser = WebScraper()
    parser.get()
    asyncio.run(write_log(message=f'finished at{dt.datetime.now()}',severity=SEVERITY.INFO))
