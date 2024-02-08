import asyncio
import datetime as dt
import glob
import os
from asyncio import as_completed
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pandasql as ps
from bs4 import BeautifulSoup

from apputils.log import write_log
from config.appconfig import SEVERITY, ZIP_FOIV, XML_STORE, RESULT_STORE


def list_inner_join(a: list, b: list):
    d = {}
    for row in b:
        d[row[0]] = row
    for row_a in a:
        row_b = d.get(row_a[0])
        if row_b is not None:  # join
            yield row_a + row_b[1:]


def loadxml(name):
    with open(name, 'r') as f:
        xml = f.read()
        try:
            #asyncio.run(write_log(message=f'file{name} at:{dt.datetime.now()}', severity=SEVERITY.INFO))
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
    df = pd.DataFrame(data=rowset, columns=['date_reg', 'workers', 'okved', 'region', 'typeface'])
    query = ("select date_reg,sum(workers) as workers,okved,region,typeface from df group by date_reg,okved,region,"
             "typeface")
    try:
        resultset = ps.sqldf(query, locals())
        resultset['date_reg']=pd.to_datetime(resultset['date_reg'])
        del df
        return resultset
    except Exception as ex:
        asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))



def debug_csv(df: pd.DataFrame):
    df.to_csv(f'{RESULT_STORE}debug.csv')


def string_to_list(i):
    while True:
        n = next(i)
        if n == '{':
            yield [x for x in string_to_list(i)]
        elif n == '}':
            return
        else:
            yield n



