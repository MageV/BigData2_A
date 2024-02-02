import asyncio
import glob
import os
import shutil
from bs4 import BeautifulSoup

from apputils.log import write_log
from config.appconfig import *


def loadxml(name):
    xml_file = open(name, 'r').read()
    try:
        soup = BeautifulSoup(xml_file, 'lxml-xml')
        return list(map(create_record, soup.find_all("Документ")))
    except Exception as ex:
        asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))


def create_record(doc) -> list:
    dat_vkl_msp = doc["ДатаВклМСП"]
    sschr = doc["ССЧР"] if "ССЧР" in doc.attrs else 0
    if "ОГРН" in doc.contents[0].attrs:
        ogrn = doc.contents[0]["ОГРН"]
    else:
        ogrn = doc.contents[0]["ОГРНИП"]
    if "ИННЮЛ" in doc.contents[0].attrs:
        inn = doc.contents[0]["ИННЮЛ"]
    else:
        inn = doc.contents[0]["ИННФЛ"]
    region_code = doc.contents[1]["КодРегион"]
    strokved = ';'.join(list(map(lambda x: x["КодОКВЭД"].split(".")[0], doc.contents[2].contents)))
    return [dat_vkl_msp, sschr, ogrn, inn, strokved, region_code]


def drop_zip():
    filelist: list = glob.glob(ZIP_FOIV + '*.zip')
    for _ in filelist:
        os.remove(_)


def drop_xml():
    filelist = glob.glob(XML_STORE + '*.xml')
    for _ in filelist:
        os.remove(_)


def drop_csv():
    filelist = glob.glob(RESULT_STORE + '*.csv')
    for _ in filelist:
        os.remove(_)


def drop_biglist():
    try:
        shutil.rmtree(BIGLIST_STORE)
        os.mkdir(BIGLIST_STORE)
    except:
        pass


def worker_reader_list(file_reader):
    rowlist = []
    for _ in file_reader:
        rowlist.append(_)
    return rowlist
