import logging
from enum import Enum, auto

import clickhouse_connect

# conn_str = 'clickhouse+asynch://default:z111111@localhost/app_storage'

DATA_HOME_FILES = '/home/master/data'
ZIP_FOIV = f'{DATA_HOME_FILES}/zips/'
XML_STORE = f'{DATA_HOME_FILES}/unpacked/'
GLOSSARY_STORE = f'{DATA_HOME_FILES}/config/json.config'
RESULT_STORE = f'{DATA_HOME_FILES}/results/'
URL_FOIV = 'https://www.nalog.gov.ru/opendata/7707329152-rsmp/'
URL_CBR = "http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx?wsdl"
MODEL_STORE = f'{DATA_HOME_FILES}/model/'

# LOG CONSTANTS
LOG_TO = 'CONSOLE'
LOG_FILE = f'{DATA_HOME_FILES}/logs/applog.log'
# MAX_DUMP_RECORDS = 10000000
# BIGLIST_STORE = f'{DATA_HOME_FILES}/biglists/'

# ANALYTIC PARAMETERS
KEY_VALUTES = ['840USD', '978EUR']

# DEBUG CONSTANTS
APP_FILE_DEBUG = True
XML_FILE_DEBUG = True
APP_FILE_DEBUG_NAME = ZIP_FOIV + 'd6d2fe2c-bf67-11ee-97b7-a20ddd0a7cd3_new.zip'
MERGE_DEBUG = True


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


class AI_FACTOR(Enum):
    AIF_KR = auto()
    AIF_USD = auto()
    AIF_EUR = auto()
    AIF_NONE = auto()

class AI_SCALER(Enum):
    AI_NONE=auto()
    AI_STD=auto()
    AI_STD_TRF=auto()

class AI_MODELS(Enum):
    AI_REGRESSORS=auto()
    AI_BEYES=auto()
    AI_ML=auto()
    AI_ALL=auto()
    AI_TREES=()

class MSP_CLASS(Enum):
    MSP_FL=0
    MSP_UL=1


DEFAULT_ARC = ARC_TYPES.ZIP
click_client = clickhouse_connect.get_client(host='localhost', database='app_storage', compress=False,
                                             username='default', password='z111111')
