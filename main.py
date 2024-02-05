import datetime as dt
import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support

from apputils.archivers import ArchiveManager
from apputils.observers import ZipFileObserver
from providers.db import ClickHouseProvider
from providers.os_operations import *
from providers.web import WebScraper

if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    archive_manager = ArchiveManager()
    db_provider = ClickHouseProvider()
    parser = WebScraper()
    parser.get_data_from_cbr()
    df = pd.DataFrame(columns=['date_', 'workers', 'okved', 'region', 'typeface', 'workers_sum'])
    processors = multiprocessing.cpu_count() - 1
    counter = 0
    total_counter = 0
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        drop_zip()
        drop_xml()
        drop_csv()
        observer = ZipFileObserver()
        store = parser.get()
        filelist = glob.glob(XML_STORE + '*.xml')
        with ProcessPoolExecutor(max_workers=processors) as pool:
            futures = [pool.submit(loadxml, item) for item in filelist]
            counter = 0
            rows = 0
            for future in as_completed(futures):
                counter += 1
                result = future.result()
                rows += result.shape[0]
                asyncio.run(
                    write_log(message=f'files processed:{counter} rows appended:{rows}', severity=SEVERITY.INFO))
                df = pd.concat([df, result], axis=0)
                del result
                gc.collect()

    elif XML_FILE_DEBUG:
        #      archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        drop_csv()
        filelist = glob.glob(XML_STORE + '*.xml')
        with ProcessPoolExecutor(max_workers=processors) as pool:
            futures = [pool.submit(loadxml, item) for item in filelist]
            counter = 0
            for future in as_completed(futures):
                counter += 1
                result = future.result()
                asyncio.run(write_log(message=f'files processed:{counter}', severity=SEVERITY.INFO))
                df = pd.concat([df, result], axis=0, ignore_index=True)
                del result
                gc.collect()
    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
"""
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

        gc.collect()
        with ProcessPoolExecutor(max_workers=processors-2) as pool:
            tasks = [pool.submit(worker_reader_list, fr) for fr in store_list.files]
            for t in tasks:
#                global_list.append(t.result())
                 df = df.append(pd.DataFrame(t.result()),ignore_index=True)
"""
