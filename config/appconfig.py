import logging
from enum import Enum, auto

from sqlalchemy.orm import declarative_base

conn_str='clickhouse://default@localhost/default'


DATA_HOME_FILES = '/home/master/data'
ZIP_FOIV = f'{DATA_HOME_FILES}/zips/'
XML_STORE = f'{DATA_HOME_FILES}/unpacked/'
JSON_STORE_CONFIG = f'{DATA_HOME_FILES}/config/json.config'
RESULT_STORE = f'{DATA_HOME_FILES}/results/'
URL = 'https://www.nalog.gov.ru/opendata/7707329152-rsmp/'
BIGLIST_STORE = f'{DATA_HOME_FILES}/biglists/'
APP_FILE_DEBUG = True
XML_FILE_DEBUG = True
APP_FILE_DEBUG_NAME = ZIP_FOIV + 'd6d2fe2c-bf67-11ee-97b7-a20ddd0a7cd3_new.zip'
LOG_TO = 'CONSOLE'
LOG_FILE = f'{DATA_HOME_FILES}/logs/applog.log'
MAX_DUMP_RECORDS = 10000000


class SEVERITY(Enum):
    INFO = logging.INFO
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG


class ARC_OPERATION(Enum):
    EXCTRACT = auto()
    PACK = auto()
    TEST = auto()


class ARC_TYPES(Enum):
    ZIP = auto()
    GZIP = auto()
    UNKWN = auto()


DEFAULT_ARC = ARC_TYPES.ZIP

Base=declarative_base()