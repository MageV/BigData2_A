import logging
from enum import Enum, auto

import clickhouse_connect

conn_str = 'clickhouse+asynch://default:z111111@localhost/app_storage'

DATA_HOME_FILES = '/home/master/data'
ZIP_FOIV = f'{DATA_HOME_FILES}/zips/'
XML_STORE = f'{DATA_HOME_FILES}/unpacked/'
GLOSSARY_STORE = f'{DATA_HOME_FILES}/config/json.config'
RESULT_STORE = f'{DATA_HOME_FILES}/results/'
URL = 'https://www.nalog.gov.ru/opendata/7707329152-rsmp/'
BIGLIST_STORE = f'{DATA_HOME_FILES}/biglists/'
APP_FILE_DEBUG = True
XML_FILE_DEBUG = True
APP_FILE_DEBUG_NAME = ZIP_FOIV + 'd6d2fe2c-bf67-11ee-97b7-a20ddd0a7cd3_new.zip'
MERGE_DEBUG=False
LOG_TO = 'CONSOLE'
LOG_FILE = f'{DATA_HOME_FILES}/logs/applog.log'
MAX_DUMP_RECORDS = 10000000
KEY_VALUTES = ['840USD', '978EUR']


class SEVERITY(Enum):
    INFO = logging.INFO
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG


class ARC_OPERATION(Enum):
    EXTRACT = auto()
    PACK = auto()
    TEST = auto()


class ARC_TYPES(Enum):
    ZIP = auto()
    GZIP = auto()
    UNKWN = auto()


DEFAULT_ARC = ARC_TYPES.ZIP
click_client = clickhouse_connect.get_client(host='localhost', database='app_storage',compress=False,
                                             username='default',password='z111111')


