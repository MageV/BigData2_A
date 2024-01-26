import asyncio

import pandas as pd
from bs4 import BeautifulSoup
from lxml import objectify
from config.appconfig import *
from apputils.log import write_log, logger


class XmlReadSingle:
    def __init__(self):
        self.__soup=None

    def loadxml(self, name):
        xml_file=open(name,'r').read()
        try:
         self.__soup=BeautifulSoup(xml_file,'xml')
         self.__parse_to_record()
         self.__drop_xml()
        except Exception as ex:
            logger.error(message=f'Error:{ex}',severity=SEVERITY.ERROR)

    def __parse_to_record(self):
        pass

    def __drop_xml(self):
        pass