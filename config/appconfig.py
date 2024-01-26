from enum import Enum, auto

ZIP_FOIV = 'data/zips/'
XML_STORE = 'data/unpacked/'
JSON_STORE_CONFIG = 'data/config/json.config'
URL = 'https://www.nalog.gov.ru/opendata/7707329152-rsmp/'
LOG_FILE = 'data/logs/applog.log'
APP_FILE_DEBUG = True
APP_FILE_DEBUG_NAME = ZIP_FOIV + '53120a20-bc2d-11ee-97b7-1e08f320e7c5_new.zip'


class SEVERITY(Enum):
    INFO = auto()
    ERROR = auto()
    DEBUG = auto()


class ARC_OPERATION(Enum):
    EXCTRACT = auto()
    PACK = auto()
    TEST = auto()


class ARC_TYPES(Enum):
    ZIP = auto()
    GZIP = auto()
    UNKWN = auto()


DEFAULT_ARC = ARC_TYPES.ZIP
