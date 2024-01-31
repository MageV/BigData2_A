import multiprocessing
import uuid
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support

from biglist import Biglist

from apputils.archivers import ArchiveManager
from apputils.observers import ZipFileObserver
from apputils.storages import ResultStorage
from providers.os_operations import *
from providers.web import WebScraper

if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    archive_manager = ArchiveManager()
    storage = ResultStorage()
    biglist_path = f'{BIGLIST_STORE}{uuid.uuid1().__str__()}/'
    biglist:Biglist = Biglist.new(path=biglist_path, batch_size=MAX_DUMP_RECORDS)
    counter = 0
    total_counter = 0
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        drop_zip()
        drop_xml()
        drop_csv()
        observer = ZipFileObserver()
        parser = WebScraper()
        store = parser.get()
        filelist = glob.glob(XML_STORE + '*.xml')
        processors = multiprocessing.cpu_count() - 1
        chunk_size = len(filelist) // processors

        with ProcessPoolExecutor(max_workers=processors, max_tasks_per_child=chunk_size) as pool:
            futures = [pool.submit(loadxml, item) for item in filelist]
            for future in as_completed(futures):
                storage.append(future.result())
    elif APP_FILE_DEBUG:
        archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
    elif XML_FILE_DEBUG:
        drop_csv()
        filelist = glob.glob(XML_STORE + '*.xml')
        processors = multiprocessing.cpu_count() - 1
        chunk_size = len(filelist) // processors
        with ProcessPoolExecutor(max_workers=processors, max_tasks_per_child=chunk_size) as pool:
            futures = [pool.submit(loadxml, item) for item in filelist]
            for future in as_completed(futures):
                biglist.append(future.result())
                counter += 1
                if counter >= MAX_DUMP_FILES_COUNT:
                    asyncio.run(write_log(message=f'dump at:{dt.datetime.now()}', severity=SEVERITY.INFO))
                    biglist.flush()
                    total_counter += counter
                    counter = 0
                    asyncio.run(write_log(message=f'total parsed:{total_counter}', severity=SEVERITY.INFO))
        biglist.flush()
    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
