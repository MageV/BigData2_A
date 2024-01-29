from bs4 import BeautifulSoup

from apputils.log import logger
from config.contexts import *
import datetime as dt


def loadxml(name):
    result = []
    xml_file = open(name, 'r').read()
    try:
        soup = BeautifulSoup(xml_file, 'xml')
        counter = file_counter_ctx.get()
        rowlist = list()
        docs = soup.find_all("Документ")
        result = list(map(create_record, docs))
        logger.info(f"{dt.datetime.now()} counter:{counter}")
        counter += 1
        file_counter_ctx.set(counter)
        # drop_xml()
    except Exception as ex:
        logger.error(msg=f'Error:{ex}')
    finally:
        return result


def drop_xml():
    pass


def create_record(doc) -> list:
    rowlist = []
    dat_vkl_msp = doc["ДатаВклМСП"]
    vid_sub_msp = doc["ВидСубМСП"]
    cat_sub_msp = doc["КатСубМСП"]
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
    rowlist.append([dat_vkl_msp, vid_sub_msp, cat_sub_msp, sschr, ogrn, inn, strokved, region_code])
    return rowlist
