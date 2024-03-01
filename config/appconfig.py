import logging
from enum import Enum, auto

import clickhouse_connect
from pyhive import hive

# conn_str = 'clickhouse+asynch://default:z111111@localhost/app_storage'

DATA_HOME_FILES = '/home/master/data'
ZIP_FOIV = f'{DATA_HOME_FILES}/zips/'
XML_STORE = f'{DATA_HOME_FILES}/unpacked/'
XLS_STORE = f'{DATA_HOME_FILES}/xlsx/'
GLOSSARY_STORE = f'{DATA_HOME_FILES}/config/json.config'
RESULT_STORE = f'{DATA_HOME_FILES}/results/'
CLASSIF_STORE = f'{DATA_HOME_FILES}/classif/'

URL_FOIV = 'https://www.nalog.gov.ru/opendata/7707329152-rsmp/'
URL_CBR_RATES = "http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx?wsdl"
URL_CBR_APP_SERVICE = "http://www.cbr.ru/CreditInfoWebServ/CreditOrgInfo.asmx?wsdl"
URL_CBR_SORS = "https://cbr.ru/statistics/bank_sector/sors/"
URL_CBR_SORS_ARC="https://cbr.ru/Queries/StatTable/Excel/302-13?lang=ru-RU"
URL_CLASSIF_OKATO = "https://rosstat.gov.ru/opendata/7708234640-7708234640-okato"
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


class PRE_TABLES(Enum):
    PT_CBR = auto()
    PT_APP = auto()
    PT_102 = auto()
    PT_SORS=auto()


class DBENGINE(Enum):
    CLICKHOUSE = auto()
    HADOOP = auto()


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



class AI_SCALER(Enum):
    AI_NONE = auto()
    AI_STD = auto()
    AI_STD_TRF = auto()


class AI_MODELS(Enum):
    AI_REGRESSORS = auto()
    AI_BEYES = auto()
    AI_ML = auto()
    AI_ALL = auto()
    AI_TREES = ()


class MSP_CLASS(Enum):
    MSP_FL = 0
    MSP_UL = 1


DEFAULT_ARC = ARC_TYPES.ZIP

topbanks = [1000, 3292, 2272, 1481, 1326, 354, 1978, 2209, 963, 2673, 3349, 3292, 2312, 2272, 650, 2590, 1, 328, 436,
            2546]  # ,2275,2268,2309,
#       1354,2168,316,429,2210,2766,2306,3255,1810,2289,3311,2998,3354,2307,2440,3252,2763,2225,415,101,705,2879,588,
#       2443,1343,2733,2170]
