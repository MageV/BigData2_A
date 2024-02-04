import gc
import multiprocessing
import uuid
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support

import pandas as pd
from biglist import Biglist
from apputils.archivers import ArchiveManager
from apputils.observers import ZipFileObserver
from providers.os_operations import *
from providers.web import WebScraper
import datetime as dt

if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    archive_manager = ArchiveManager()
    drop_biglist()
    global_list = []
    df=pd.DataFrame()
    processors = multiprocessing.cpu_count() - 1
    biglist_path = f'{BIGLIST_STORE}{uuid.uuid1().__str__()}'
    store_list = Biglist.new(path=biglist_path, batch_size=MAX_DUMP_RECORDS)
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
        with ProcessPoolExecutor(max_workers=processors) as pool:
            futures = [pool.submit(loadxml, item) for item in filelist]
            filecount = 0
            for future in as_completed(futures):
                filecount += 1
                store_list.append(future.result())
                if filecount > 100:
                    total_counter += filecount
                    store_list.flush()
                    filecount = 0
                    gc.collect()
                    asyncio.run(
                        write_log(
                            message=f'dump at:{dt.datetime.now()} total parsed:{total_counter} of {len(filelist)}',
                            severity=SEVERITY.INFO))
        store_list.flush()
        with ProcessPoolExecutor(max_workers=processors) as pool:
            tasks = [pool.submit(worker_reader_list, fr) for fr in store_list.files]
            for t in tasks:
                global_list.append(t.result())
    elif XML_FILE_DEBUG:
        archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        drop_csv()
        filelist = glob.glob(XML_STORE + '*.xml')
        with ProcessPoolExecutor(max_workers=processors) as pool:
            futures = [pool.submit(loadxml, item) for item in filelist]
            filecount = 0
            for future in as_completed(futures):
                filecount += 1
                store_list.append(future.result())
                if filecount > 100:
                    total_counter += filecount
                    store_list.flush()
                    filecount = 0
                    gc.collect()
                    asyncio.run(
                        write_log(
                            message=f'dump at:{dt.datetime.now()} total parsed:{total_counter} of {len(filelist)}',
                            severity=SEVERITY.INFO))

        store_list.flush()
        with ProcessPoolExecutor(max_workers=processors) as pool:
            tasks = [pool.submit(worker_reader_list, fr) for fr in store_list.files]
            for t in tasks:
#                global_list.append(t.result())
                 df = df.append(pd.DataFrame(t.result()),ignore_index=True)
    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
