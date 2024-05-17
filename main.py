import gc
import glob
import multiprocessing
from multiprocessing import freeze_support
import warnings

from apputils.archivers import ArchiveManager
from apputils.observers import ZipFileObserver
from apputils.utils import drop_zip, drop_xml, drop_csv, drop_xlsx, preprocess_xml
from ml.mod_tflow import tf_learn_model
from providers.df import *
from providers.web import WebScraper

warnings.filterwarnings("ignore")


def app_init():
    a_manager = ArchiveManager()
    webparser = WebScraper()
    dbprovider = DBConnector()
    prc = 1 if (multiprocessing.cpu_count() - 4) == 0 else multiprocessing.cpu_count() - 4
    frame = pd.DataFrame(columns=['date_', 'workers', 'region', 'typeface', 'workers_sum'])  # 'okved',
    return a_manager, webparser, prc, frame, dbprovider


if __name__ == '__main__':
    freeze_support()
    asyncio.run(write_log(message=f'Started at:{dt.datetime.now()}', severity=SEVERITY.INFO))
    archive_manager, parser, processors, df, dbprovider = app_init()
    filelist = glob.glob(XML_STORE + '*.xml')
    if not DISABLE_LOAD:
        okato = parser.get_regions()
        credit_msp = parser.get_sors(processors_count=processors)
        credit_arc_msp = parser.get_sors_archive()
        debt_msp = parser.get_debt(processors_count=processors)
        debt_arc_msp = parser.get_debt_arc()
        msp_sors = pd.concat([credit_msp, credit_arc_msp], axis=0, ignore_index=True)
        msp_debt = pd.concat([debt_msp, debt_arc_msp], axis=0, ignore_index=True)
        dbprovider.db_write_okato(okato)
        dbprovider.db_prepare_tables(PRE_TABLES.PT_SORS)
        dbprovider.db_write_credit_info(okato, msp_sors, PRE_TABLES.PT_SORS)
        dbprovider.db_prepare_tables(PRE_TABLES.PT_DEBT)
        dbprovider.db_write_credit_info(okato, msp_debt, PRE_TABLES.PT_DEBT)
    if not APP_FILE_DEBUG and not XML_FILE_DEBUG:
        drop_zip()
        drop_xml()
        drop_csv()
        drop_xlsx()
        observer = ZipFileObserver()
        store_fns = parser.get_FNS(url=URL_FOIV)
        archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        dbprovider.db_prepare_tables(PRE_TABLES.PT_APP)
        df = preprocess_xml(file_list=filelist, processors_count=processors, db_provider=dbprovider)
        okatos = dbprovider.get_unq_okatos()
        regdates = dbprovider.db_get_minmax()
        sors = dbprovider.get_credit_info()
        app = dbprovider.db_get_frames_by_facetype(ft=MSP_CLASS.MSP_UL.value)
        asyncio.run(write_log(message=f'Start for UL:{dt.datetime.now()}', severity=SEVERITY.INFO))
        raw_data = df_fill_credit_apps(typeface=MSP_CLASS.MSP_UL, sors_frame=sors,
                                       app_frame=app, dates_frame=regdates)
        app = dbprovider.db_get_frames_by_facetype(ft=MSP_CLASS.MSP_FL.value)
        asyncio.run(write_log(message=f'Start for FL:{dt.datetime.now()}', severity=SEVERITY.INFO))
        raw_data_2 = df_fill_credit_apps(typeface=MSP_CLASS.MSP_FL, sors_frame=sors,
                                         app_frame=app, dates_frame=regdates)
        raw_data_total = pd.concat([raw_data, raw_data_2], axis=0, ignore_index=True)
        asyncio.run(write_log(message=f'Finish for app_rows:FL:{dt.datetime.now()}', severity=SEVERITY.INFO))
        gc.collect()

    elif XML_FILE_DEBUG:
        #      archive_manager.extract(source=APP_FILE_DEBUG_NAME, dest=XML_STORE)
        files_csv = glob.glob(RESULT_STORE + '*.csv')
        if not MERGE_DEBUG:
            dbprovider.db_prepare_tables(PRE_TABLES.PT_APP)
            df = preprocess_xml(file_list=filelist, processors_count=processors, db_provider=dbprovider)
        else:
            df = preprocess_xml(file_list=filelist, processors_count=processors, debug=True, db_provider=dbprovider)
        # TO COPY IN NON-DEBUG part
        okatos = dbprovider.get_unq_okatos()
        regdates = dbprovider.db_get_minmax()
        sors = dbprovider.get_credit_info(PRE_TABLES.PT_SORS)
        debt = dbprovider.get_credit_info(PRE_TABLES.PT_DEBT)
        app = dbprovider.db_get_frames_by_facetype(ft=MSP_CLASS.MSP_UL.value)
        app=df_remove_outliers(app,okatos,"sworkers")
        asyncio.run(write_log(message=f'Merge DF for UL:{dt.datetime.now()}', severity=SEVERITY.INFO))
        raw_data = df_fill_credit_apps(typeface=MSP_CLASS.MSP_UL, sors_frame=sors, debt_frame=debt,
                                       app_frame=app, dates_frame=regdates)
        app = dbprovider.db_get_frames_by_facetype(ft=MSP_CLASS.MSP_FL.value)
        app = df_remove_outliers(app, okatos, "sworkers")
        asyncio.run(write_log(message=f'Merge DF for FL:{dt.datetime.now()}', severity=SEVERITY.INFO))
        raw_data_2 = df_fill_credit_apps(typeface=MSP_CLASS.MSP_FL, sors_frame=sors, debt_frame=debt,
                                         app_frame=app, dates_frame=regdates)
        raw_data_total = pd.concat([raw_data, raw_data_2], axis=0, ignore_index=True)
        asyncio.run(write_log(message=f'Finish for app_rows:FL:{dt.datetime.now()}', severity=SEVERITY.INFO))
        gc.collect()
        mclass_data, boundaries, labels = df_create_raw_data(db_provider=dbprovider, appframe=raw_data_total,
                                                             is_multiclass=True)
        binary_data = df_create_raw_data(db_provider=dbprovider, appframe=raw_data_total, is_multiclass=False)
        tf_learn_model(binary_data, 0.25, 0.15, TF_OPTIONS.TF_LSTM)
        #sk_learn_model([mclass_data, boundaries, labels], features=None,
        #               models_class=AI_MODELS.AI_REGRESSORS, is_multiclass=True)
        #sk_learn_model(binary_data, features=None, models_class=AI_MODELS.AI_CLASSIFIERS, is_multiclass=False)
        #tf_learn_model([mclass_data, boundaries, labels], pct_val=0.20, pct_train=0.15,
        #              classifier=TF_OPTIONS.TF_NN_MULTU)
        # tf_learn_model(binary_data, pct_val=0.20,pct_train=0.15, classifier=TF_OPTIONS.TF_NN_BINARY)
        #   tf_learn_model(binary_data, 0.15, 0.1, TF_OPTIONS.TF_TREES_BINARY)
        # tf_learn_model(binary_data,0.15,0.1,TF_OPTIONS.TF_NN_BINARY)
    asyncio.run(write_log(message=f'finished at:{dt.datetime.now()}', severity=SEVERITY.INFO))
