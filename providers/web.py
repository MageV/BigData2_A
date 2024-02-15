import asyncio
import datetime as dt
from uuid import uuid1
from zeep import Client, Settings
import aiofiles
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer

from apputils.log import write_log
from apputils.utils import list_inner_join
from config.appconfig import *


class WebScraper:

    def __init__(self, url=''):
        self.__headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en"
        }
        if url == '':
            self.__url = URL_FOIV
        self.__html = ""

    async def __getHtml(self):
        async  with aiohttp.ClientSession() as session:
            response = await session.get(self.__url, headers=self.__headers)
            self.__html = await response.text()

    def __parseHtml(self):
        soup = BeautifulSoup(self.__html, features="lxml")
        files = []
        tables_html = soup.find('table').find_all('a')
        for _ in tables_html:
            if _.text.__contains__('zip'):
                files.append(_.text.split('-'))
        df = pd.DataFrame(files, columns=['head_zip', 'subpath', 'actual_data', 'service', 'tail_zip'])
        try:
            df = df.assign(fmtDate=pd.to_datetime(df['actual_data'], format='%d%M%Y', dayfirst=True))
            top: pd.DataFrame = df.sort_values(by='fmtDate', ascending=False).head(1)
            top.to_json(GLOSSARY_STORE, orient='records', lines=True)
            del top['fmtDate']
            filename = top.to_string(header=False, index=False, index_names=False).replace(' ', '-')[:-1]
            return filename
        except Exception as ex:
            print(ex)
        return None

    async def __get_file(self, name, store):
        # chunk_size = 64*1024*1024
        chunk_size = 8192
        await write_log(message=f'Zip download started at: {dt.datetime.now()}', severity=SEVERITY.INFO)
        timeout = aiohttp.ClientTimeout(total=60 * 60, sock_read=240)
        async with aiohttp.ClientSession(timeout=timeout).get(name) as response:
            async with aiofiles.open(store, mode="wb") as f:
                while True:
                    chunk = await response.content.read(chunk_size)
                    await asyncio.sleep(0)
                    if not chunk:
                        break
                    await f.write(chunk)
        await write_log(message=f'Zip download completed at: {dt.datetime.now()}', severity=SEVERITY.INFO)

    def get(self):
        asyncio.run(self.__getHtml())
        file = self.__parseHtml()
        store = f'{ZIP_FOIV}{str(uuid1())}_new.zip'
        asyncio.run(self.__get_file(name=file, store=store))
        return store

    def get_data_from_cbr(self, mindate=dt.datetime.strptime('01.01.2010', '%d.%m.%Y'), maxdate=dt.datetime.today()):
        client = Client(URL_CBR)
        client.settings = Settings(raw_response=True, strict=False)
        response_key_rate = client.service.KeyRate(mindate, maxdate)
        strainer_rate = SoupStrainer("KR")
        key_rates_spr = list()
        result_list = list()
        soup_key_rate = BeautifulSoup(response_key_rate.content, 'lxml-xml', parse_only=strainer_rate)
        for item in soup_key_rate.contents:
            dt_value = dt.datetime.strptime(item.find_next('DT').text.split('T')[0], format('%Y-%m-%d'))
            key_value = float(item.find_next('Rate').text)
            key_rates_spr.append((dt_value, key_value))
 #       response_val_spr = client.service.EnumValutes(False)
 #       soup_val_spr = BeautifulSoup(response_val_spr.content, 'lxml-xml')
 #       valutes = soup_val_spr.find_all("EnumValutes")
 #       for item in valutes:
 #           temp_list = list(filter(None, item.text.split(' ')))
 #           if temp_list[-1:][0] in KEY_VALUTES:
 #               request = temp_list[0]
 #               soup = BeautifulSoup((client.service.GetCursDynamic(mindate, maxdate, request)).content, "lxml-xml")
 #               val_rowset = list()
 #               for item in soup.find_all("ValuteCursDynamic"):
 #                   date_val_date = dt.datetime.strptime(item.text.split('T')[0], format('%Y-%m-%d'))
 ##                   curs_val_date = float(item.find_next("Vcurs").text)
  #                  if date_val_date.day == 1:
  #                      val_rowset.append((date_val_date, curs_val_date))
  #              result_list.append(list(list_inner_join(key_rates_spr, val_rowset)))
        df=pd.DataFrame(key_rates_spr)
#        for _ in result_list:
#            frame=pd.DataFrame(_)
#            df=pd.concat([df,frame],ignore_index=True,axis=1)
#        df.drop(df.columns[[3,4]],axis=1,inplace=True)
        df.columns=["date_","key_rate"]#,"val_usd","val_eur"]
        #df.reset_index(inplace=True)
        return df
