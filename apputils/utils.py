import asyncio
import datetime as dt
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
import scipy.stats as st

from apputils.log import write_log
from config.appconfig import *
from pathlib import Path


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
            soup = BeautifulSoup(xml, 'lxml-xml')
            return list2df(list(map(create_record, soup.find_all("Документ"))))
        except Exception as ex:
            asyncio.run(write_log(message=f'Error:{ex}', severity=SEVERITY.ERROR))


def create_record(doc) -> list:
    dat_vkl_msp = dt.datetime.strptime(doc["ДатаВклМСП"], '%d.%m.%Y')
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
    #  strokved = ';'.join(set(map(lambda x: x["КодОКВЭД"].split(".")[0], doc.contents[2].contents)))
    return [dat_vkl_msp, sschr, region_code, typeface]  # strokved,


def create_record_v2(doc) -> list:
    dat_vkl_msp = dt.datetime.strptime(doc["ДатаВклМСП"], '%d.%m.%Y')
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
    strokved = set(map(lambda x: x["КодОКВЭД"].split(".")[0], doc.contents[2].contents))
    retlist = [[dat_vkl_msp, sschr, x, region_code, typeface] for x in strokved]
    return retlist


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


def drop_xlsx():
    filelist = glob.glob(XLS_STORE + '*.*')
    for _ in filelist:
        os.remove(_)


def date_to_beg_month(in_date: dt.datetime):
    month = in_date.month
    year = in_date.year
    day = 1
    return dt.datetime(year=year, month=month, day=day)


def list2df(rowset: list):
    df = pd.DataFrame(data=rowset, columns=['date_reg', 'workers', 'region', 'typeface'])  # 'okved',
    df['date_reg'] = df['date_reg'].apply(lambda x: date_to_beg_month(x))
    df['date_reg'] = pd.to_datetime(df['date_reg'])
    return df


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


def storage_init():
    Path(DATA_HOME_FILES).mkdir(parents=True, exist_ok=True)
    Path(ZIP_FOIV).mkdir(parents=True, exist_ok=True)
    Path(XLS_STORE).mkdir(parents=True, exist_ok=True)
    Path(XML_STORE).mkdir(parents=True, exist_ok=True)
    Path(RESULT_STORE).mkdir(parents=True, exist_ok=True)
    Path(CLASSIF_STORE).mkdir(parents=True, exist_ok=True)
    Path(MODEL_STORE).mkdir(parents=True, exist_ok=True)
    Path(LOG_STORE).mkdir(parents=True, exist_ok=True)


def multiclass_binning(frame, col_name, classes=9):
    binned = 'estimated'
    labels = [_ for _ in range(classes)]
    frame[binned], boundaries = pd.qcut(frame[col_name], q=classes, precision=1, retbins=True, labels=labels)
    return frame, boundaries, labels


def detect_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for " + dist_name + " = " + str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: " + str(best_dist))
    print("Best p value: " + str(best_p))
    print("Parameters for the best fit: " + str(params[best_dist]))
    return best_dist, best_p, params[best_dist]


def prepare_con_mat(con_mat, classes):
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)
    return con_mat_df
