import asyncio
import datetime as dt
import gc
import glob
import os
import shutil

from bs4 import BeautifulSoup

from apputils.log import write_log
from config.appconfig import *


def loadxml(name):
    xml_file = open(name, 'r').read()
    result = []
    try:
        soup = BeautifulSoup(xml_file, 'lxml-xml')
        docs = soup.find_all("Документ")
        result = list(map(create_record, docs))
        soup.clear()
        gc.collect()
        # asyncio.run(write_log(message=f"{dt.datetime.now()}:{name}", severity=SEVERITY.INFO))
    except Exception as ex:
        asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))
    finally:
        return result


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
    shutil.rmtree(BIGLIST_STORE)
    os.mkdir(BIGLIST_STORE)
