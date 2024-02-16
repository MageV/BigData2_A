import gc
import glob
import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support
import warnings

from apputils.archivers import ArchiveManager
from apputils.observers import ZipFileObserver
from apputils.utils import loadxml, drop_zip, drop_xml, drop_csv
from ml.ai_model import ai_learn
from providers.df import *
from providers.web import WebScraper

warnings.filterwarnings("ignore")


def app_init():
    a_manager = ArchiveManager()
    webparser = WebScraper()
    webparser.get_top_50()
    dbpovider = DBConnector()
    prc = 1 if (multiprocessing.cpu_count() - 4) == 0 else multiprocessing.cpu_count() - 4
    frame = pd.DataFrame(columns=['date_', 'workers', 'okved', 'region', 'typeface', 'workers_sum'])
    return a_manager, webparser, prc, frame, dbpovider


def preprocess_xml(file_list, processors_count, db_provider, debug=False):
    if debug:
        result = db_provider.get_minmax()
        return result.result_rows
    big_frame = pd.DataFrame(columns=['date_reg', 'workers', 'okved', 'region', 'typeface'])
    asyncio.run(write_log(message=f'Parse started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    with (ProcessPoolExecutor(max_workers=processors_count,
                              max_tasks_per_child=len(file_list) // processors_count + 20) as pool):
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
            #            click_client.insert_df(table='app_row', df=big_frame, column_names=['date_reg', 'workers', 'okved',
            #                                                                                'region', 'typeface', 'ratekey',
            #                                                                                'usd',
            #                                                                               'eur'],
            #                                   column_type_names=['Date', 'Int32', 'String', 'Int32', 'Int32', 'Float32',
            #                                                      'Float32', 'Float32'], settings=settings)
            db_provider.insert_data(big_frame)
            asyncio.run(write_log(message=f'Success to store data:{dt.datetime.now()}',
                                  severity=SEVERITY.INFO))
        except Exception as ex:
            asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))
        result = db_provider.get_minmax()
        return result


def prepare_f102(scraper: WebScraper):
    pass


if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    archive_manager, parser, processors, df, dbprovider = app_init()
    prepare_f102(scraper=parser)
    filelist = glob.glob(XML_STORE + '*.xml')
    counter = 0
    total_counter = 0
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        drop_zip()
        drop_xml()
        drop_csv()
        observer = ZipFileObserver()
        store_fns = parser.get_FNS(url=URL_FOIV)
        archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        dbprovider.db_prepare_tables('app')
        df = preprocess_xml(file_list=filelist, processors_count=processors)
        kvframe = dbprovider.fill_glossary(parser, df[0][0], df[0][1])
        asyncio.run(write_log(message=f'Update app_row:Started:{dt.datetime.now()}', severity=SEVERITY.INFO))
        dbprovider.update_rows_kv(kvframe)
        asyncio.run(write_log(message=f'Update app_row:finished:{dt.datetime.now()}', severity=SEVERITY.INFO))
        gc.collect()
        ai_learn(AI_FACTOR.AIF_KR,db_provider=dbprovider, scaler=AI_SCALER.AI_STD_TRF, models_class=AI_MODELS.AI_TREES,
                 msp_class=MSP_CLASS.MSP_UL)
    elif XML_FILE_DEBUG:
        #      archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        files_csv = glob.glob(RESULT_STORE + '*.csv')
        # drop_csv()
        if not MERGE_DEBUG:
            dbprovider.db_prepare_tables('app')
            df = preprocess_xml(file_list=filelist, processors_count=processors,db_provider=dbprovider)
        else:
            df = preprocess_xml(file_list=filelist, processors_count=processors, debug=True,db_provider=dbprovider)
        kvframe = dbprovider.fill_glossary(parser, df[0][0], df[0][1])
        asyncio.run(write_log(message=f'Update app_row:Started:{dt.datetime.now()}', severity=SEVERITY.INFO))
        dbprovider.update_rows_kv(kvframe)
        asyncio.run(write_log(message=f'Update app_row:finished:{dt.datetime.now()}', severity=SEVERITY.INFO))
        gc.collect()
        ai_learn(AI_FACTOR.AIF_KR,db_provider=dbprovider, scaler=AI_SCALER.AI_STD_TRF, models_class=AI_MODELS.AI_TREES,
                 msp_class=MSP_CLASS.MSP_UL)
    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
