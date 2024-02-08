import asyncio
import gc
import glob
import multiprocessing
import datetime as dt
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support

import pandas as pd
from joblib import Parallel

from apputils.archivers import ArchiveManager
from apputils.log import write_log
from apputils.observers import ZipFileObserver
from apputils.utils import loadxml, drop_zip, drop_xml, drop_csv, debug_csv
from providers.db import *
from providers.web import WebScraper


def app_init():
    a_manager = ArchiveManager()
    webparser = WebScraper()
    prc = 1 if (multiprocessing.cpu_count() - 4) == 0 else multiprocessing.cpu_count() - 4
    frame = pd.DataFrame(columns=['date_', 'workers', 'okved', 'region', 'typeface', 'workers_sum'])
    return a_manager, webparser, prc, frame


def preprocess_xml(file_list, processors_count):
    big_frame = pd.DataFrame(columns=['date_reg', 'workers', 'okved', 'region', 'typeface'])
    asyncio.run(write_log(message=f'Parse started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    with (ProcessPoolExecutor(max_workers=processors_count,
                              max_tasks_per_child=len(file_list) // processors_count+20) as pool):
        futures = [pool.submit(loadxml, item) for item in file_list]
        row_counter = 0
        for future in as_completed(futures):
            row_counter += 1
            result = future.result()
            big_frame = pd.concat([big_frame, result], axis=0, ignore_index=True)
            asyncio.run(write_log(message=f'files processed:{row_counter} at {dt.datetime.now()}',
                                  severity=SEVERITY.INFO))
            del result
        try:
            big_frame['ratekey'] = 0.0
            big_frame['usd'] = 0.0
            big_frame['eur'] = 0.0
            settings = {'async_insert': 1}
            asyncio.run(write_log(message=f'Trying to store data:{dt.datetime.now()}',
                                  severity=SEVERITY.INFO))
            click_client.insert_df(table='app_row', df=big_frame, column_names=['date_reg', 'workers', 'okved',
                                                                                'region', 'typeface', 'ratekey',
                                                                                'usd',
                                                                                'eur'],
                                   column_type_names=['Date', 'Int32', 'String', 'Int32', 'Int32', 'Float32',
                                                      'Float32', 'Float32'], settings=settings)
            asyncio.run(write_log(message=f'Success to store data:{dt.datetime.now()}',
                                  severity=SEVERITY.INFO))
        except Exception as ex:
            asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))
        big_frame.drop()



if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    archive_manager, parser, processors, df = app_init()
    asyncio.run(write_log(message=f'Load data from CBR:{dt.datetime.now()}', severity=SEVERITY.INFO))
    kv_dframe = parser.get_data_from_cbr()
    prepare_tables('cbr')
    asyncio.run(
        write_log(message=f'Glossary:Write data to ClickHouse started:{dt.datetime.now()}', severity=SEVERITY.INFO))
    click_client.insert_df(table='app_cbr', df=kv_dframe, column_names=['date_', 'keyrate', 'usd', 'eur'],
                           column_type_names=['Date', 'Float32', 'Float32', 'Float32'])
    asyncio.run(write_log(message=f'Glossary:Finished:{dt.datetime.now()}', severity=SEVERITY.INFO))
    del kv_dframe
    filelist = glob.glob(XML_STORE + '*.xml')
    counter = 0
    total_counter = 0
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        drop_zip()
        drop_xml()
        drop_csv()
        observer = ZipFileObserver()
        store = parser.get()
        archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        prepare_tables('app')
        df = preprocess_xml(file_list=filelist, processors_count=processors, debug=False)
    elif XML_FILE_DEBUG:
        #      archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        files_csv = glob.glob(RESULT_STORE + '*.csv')
        # drop_csv()
        if MERGE_DEBUG:
            prepare_tables('app')
            df = preprocess_xml(file_list=filelist, processors_count=processors)
        else:
            gc.collect()
            val_df = prepare_ml_data()
    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
