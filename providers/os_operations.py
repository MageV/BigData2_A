import asyncio
import gc
import glob
import os
import shutil
import pandasql as ps

import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import datetime as dt

from apputils.log import write_log
from config.appconfig import *


def loadxml(name):
    with open(name, 'r') as f:
        xml = f.read()
        try:
            soup = BeautifulSoup(xml, 'lxml-xml')
            return reducer(list(map(create_record, soup.find_all("Документ"))))
        except Exception as ex:
            asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))


def create_record(doc) -> list:
    dat_vkl_msp = dt.datetime.strptime(doc["ДатаВклМСП"], '%d.%m.%Y')
    dat_vkl_msp = dat_vkl_msp.replace(day=1)
    sschr = int(doc["ССЧР"]) if "ССЧР" in doc.attrs else 1
    #    if "ОГРН" in doc.contents[0].attrs:
    #        ogrn = doc.contents[0]["ОГРН"]
    #    else:
    #       ogrn = doc.contents[0]["ОГРНИП"]
    if "ИННЮЛ" in doc.contents[0].attrs:
        # inn = doc.contents[0]["ИННЮЛ"]
        typeface = 1
    else:
        typeface = 0
    # inn = doc.contents[0]["ИННФЛ"]
    region_code = int(doc.contents[1]["КодРегион"])
    strokved = ';'.join(set(map(lambda x: x["КодОКВЭД"].split(".")[0], doc.contents[2].contents)))
    return [dat_vkl_msp, sschr, strokved, region_code, typeface]


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


def reducer(rowset: list):
    df = pd.DataFrame(data=rowset, columns=['date_', 'workers', 'okved', 'region', 'typeface'])
    query = "select *,sum(workers) as workers_sum from df group by date_,okved,region,typeface"
    try:
        resultset = ps.sqldf(query, locals())
        del df
        return resultset
    except Exception as ex:
        asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))
