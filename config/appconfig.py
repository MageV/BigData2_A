from enum import Enum, auto

ZIP_FOIV='data/zips/'
XML_STORE='data/unpacked/'
JSON_STORE_CONFIG='data/config/json.config'
URL='https://www.nalog.gov.ru/opendata/7707329152-rsmp/'
LOG_FILE='data/logs/applog.log'

class SEVERITY(Enum):
    INFO=auto()
    ERROR=auto()
    DEBUG=auto()