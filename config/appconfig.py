import logging
from enum import Enum, auto

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



# DEBUG CONSTANTS
APP_FILE_DEBUG = True
XML_FILE_DEBUG = True
APP_FILE_DEBUG_NAME = ZIP_FOIV + 'd6d2fe2c-bf67-11ee-97b7-a20ddd0a7cd3_new.zip'
MERGE_DEBUG = True
CREDITS_DEBUG=True


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
    AI_REGRESSORS = "_regressors_"
    AI_BEYES = "_beyes_"
    AI_ML = "_svc_svr_"
    AI_ALL = "_all_"
    AI_CLASSIFIERS = "_classifiers_"
    AI_EXPERIMENTAL="_experimental_"


class MSP_CLASS(Enum):
    MSP_FL = 0
    MSP_UL = 1


DEFAULT_ARC = ARC_TYPES.ZIP


