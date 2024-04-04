import asyncio
import datetime as dt
import glob
import os.path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from uuid import uuid1
import aiofiles
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from zeep import Client, Settings

from apputils.log import write_log
from config.appconfig import *

warnings.filterwarnings("ignore")


class WebScraper:

    def __init__(self):
        self.__headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/70.0.3538.110 Safari/537.36",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en"
        }

    async def __getHtml(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.__headers) as resp:
                #await session.close()
                return await resp.text()






    def __parseHtml_FNS(self, html):
        soup = BeautifulSoup(html, features="lxml")
        files = []
        tables_html = soup.find('table').find_all('a')
        for _ in tables_html:
            if _.text.__contains__('zip'):
                files.append(_.text.split('-'))
        df = pd.DataFrame(files, columns=['head_zip', 'subpath', 'actual_data', 'service', 'tail_zip'])
        try:
            df = df.assign(fmtDate=pd.to_datetime(df['actual_data'], format='%d%M%Y', dayfirst=True))
            top: pd.DataFrame = df.sort_values(by='fmtDate', ascending=False).head(1)
            # top.to_json(GLOSSARY_STORE, orient='records', lines=True)
            del top['fmtDate']
            filename = top.to_string(header=False, index=False, index_names=False).replace(' ', '-')[:-1]
            return filename
        except Exception as ex:
            print(ex)
        return None

    async def __get_file_web(self, name, store):
        # chunk_size = 64*1024*1024
        chunk_size = 8192
        await write_log(message=f'File download started at: {dt.datetime.now()}', severity=SEVERITY.INFO)
        timeout = aiohttp.ClientTimeout(total=60 * 60, sock_read=240)
        async with aiohttp.ClientSession(timeout=timeout, headers=self.__headers).get(name) as response:
            async with aiofiles.open(store, mode="wb") as f:
                while True:
                    chunk = await response.content.read(chunk_size)
                    await asyncio.sleep(0)
                    if not chunk:
                        break
                    await f.write(chunk)
        await write_log(message=f'File download completed at: {dt.datetime.now()}', severity=SEVERITY.INFO)

    def get_FNS(self, url=''):
        html = asyncio.run(self.__getHtml(url))
        file = self.__parseHtml_FNS(html=html)
        store = f'{ZIP_FOIV}{str(uuid1())}_new.zip'
        asyncio.run(self.__get_file_web(name=file, store=store))
        return store

    def get_rates_cbr(self, mindate=dt.datetime.strptime('01.01.2010', '%d.%m.%Y'), maxdate=dt.datetime.today()):
        client = Client(URL_CBR_RATES)
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
        df = pd.DataFrame(key_rates_spr)
        df.columns = ["date_", "key_rate"]  # ,"val_usd","val_eur"]
        return df

    def get_regions(self) -> pd.DataFrame:
        if not os.path.isfile(f"{CLASSIF_STORE}okato.csv"):
            html = asyncio.run(self.__getHtml(URL_CLASSIF_OKATO))
            soup = BeautifulSoup(html, "html.parser").find_all("a", href=True, class_="btn-sm")
            for item in soup:
                if item.attrs['href'].__contains__("data-"):
                    asyncio.run(self.__get_file_web(item.attrs['href'], f"{CLASSIF_STORE}okato.csv"))

        csv_set = pd.read_csv(f"{CLASSIF_STORE}okato.csv", encoding='cp1251', delimiter=";")
        csv_set.columns = ["okato_0", "okato_1", "okato_2", "okato_3", "okato_4", "regname", "capital", "okato_7",
                           "okato_8", "okato_9", "date_1", "date_2"]
        csv_set = csv_set[(csv_set["okato_1"] == 0) & (csv_set["okato_4"] == 1)]
        okato = csv_set[["okato_0", "regname"]]
        return okato
        pass



    async def __import_sors_list(self):
        html = await self.__getHtml(URL_CBR_SORS)
        soup = BeautifulSoup(html, features="lxml").find_all("a", class_="versions_item")
        list_cred = list()
        for item in soup:
            if item.attrs['href'].__contains__("loans_sme_branches"):
                list_cred.append("https://cbr.ru" + item.attrs['href'])
        return list_cred

    async def __import_debt_list(self):
        html=await self.__getHtml(URL_CBR_SORS)
        soup = BeautifulSoup(html, features="lxml").find_all("a", class_="versions_item")
        list_debt = list()
        for item in soup:
            if item.attrs['href'].__contains__("Debt_sme_divisions"):
                list_debt.append("https://cbr.ru" + item.attrs['href'])
        return list_debt


    def _load_xlsx_sors(self, name):
        frame = pd.read_excel(name, engine="openpyxl", header=list(range(6)))
        frame_date_str = name.split('_')[-1].split('.')[0]
        frame_date = dt.datetime.strptime(frame_date_str, "%Y%m%d")
        frame = frame.iloc[:, :4]
        frame.columns = ["region", "total", "msp_total", "il_total"]
        frame["msp_total"] -= frame["il_total"]
        frame["date_rep"] = frame_date
        return frame

    def _load_xlsx_debt(self, name):
        frame = pd.read_excel(name, engine="openpyxl", header=list(range(6)))
        frame_date_str = name.split('_')[-1].split('.')[0]
        frame_date = dt.datetime.strptime(frame_date_str, "%Y%m%d")
        frame = frame.iloc[:, :4]
        frame.columns = ["region", "total", "msp_total", "il_total"]
        frame["msp_total"] -= frame["il_total"]
        frame["date_rep"] = frame_date
        return frame

    # TO-DO
    def get_sors_archive(self):
        asyncio.run(self.__get_file_web(URL_CBR_SORS_ARC, f"{XLS_STORE}sors_arc.xlsx"))
        xl = pd.ExcelFile(f"{XLS_STORE}sors_arc.xlsx")
        frame_result = pd.DataFrame(columns=["region", "total", "msp_total", "il_total"])
        for name in xl.sheet_names:
            frame = xl.parse(name, header=list(range(8)))
            frame = frame.iloc[:, :4]
            frame.columns = ["region", "total", "msp_total", "il_total"]
            frame["msp_total"] -= frame["il_total"]
            date_sheet = dt.datetime.strptime((name.split(' '))[-1], format("%d.%m.%Y"))
            frame['date_rep'] = date_sheet
            frame_result = pd.concat([frame_result, frame], axis=0, ignore_index=True)
        frame_result = frame_result[
            (frame_result['region'].str.contains('ОКРУГ') == False) & (frame_result['region'].str[0] != ' ')]
        frame_result = frame_result[(frame_result['region'].str.contains('округ') == False)]
        frame_result["okato_code"] = 0
        return frame_result

    def get_sors(self, processors_count) -> pd.DataFrame:
        file_list = asyncio.run(self.__import_sors_list())
        for item in file_list:
            name_xlsx = item.split('/')[-1]
            asyncio.run(self.__get_file_web(item, XLS_STORE + "sors_oper_" + name_xlsx))
            asyncio.sleep(0.5)
        xls_list = glob.glob(XLS_STORE + 'sors_oper_*.xlsx')
        frame_result = pd.DataFrame(columns=["region", "total", "msp_total", "il_total"])
        with (ProcessPoolExecutor(max_workers=processors_count,
                                  max_tasks_per_child=len(xls_list) // processors_count + 20) as pool):
            futures = [pool.submit(self._load_xlsx_sors, item) for item in xls_list]
            for future in as_completed(futures):
                try:
                    frame_result = pd.concat([frame_result, future.result()], axis=0, ignore_index=True)
                except ValueError as ex:
                    break
        frame_result = frame_result[
            (frame_result['region'].str.contains('ОКРУГ') == False) & (frame_result['region'].str[0] != ' ')&(frame_result['region'].str.contains('ФЕДЕРАЦИЯ') == False)]
        frame_result["okato_code"] = 0
        return frame_result

    def get_debt(self, processors_count) -> pd.DataFrame:
        file_list = asyncio.run(self.__import_debt_list())
        for item in file_list:
            name_xlsx = item.split('/')[-1]
            asyncio.run(self.__get_file_web(item, XLS_STORE + "debt_oper_" + name_xlsx))
            asyncio.sleep(0.5)
        xls_list = glob.glob(XLS_STORE + 'debt_oper_*.xlsx')
        frame_result = pd.DataFrame(columns=["region", "total", "msp_total", "il_total"])
        with (ProcessPoolExecutor(max_workers=processors_count,
                                  max_tasks_per_child=len(xls_list) // processors_count + 20) as pool):
            futures = [pool.submit(self._load_xlsx_debt, item) for item in xls_list]
            for future in as_completed(futures):
                try:
                    frame_result = pd.concat([frame_result, future.result()], axis=0, ignore_index=True)
                except ValueError as ex:
                    break
        frame_result = frame_result[
            (frame_result['region'].str.contains('ОКРУГ') == False) & (frame_result['region'].str[0] != ' ')&(frame_result['region'].str.contains('ФЕДЕРАЦИЯ') == False)]
        frame_result["okato_code"] = 0
        return frame_result

    def get_debt_arc(self):
        asyncio.run(self.__get_file_web(URL_CBR_DEBT_ARC, f"{XLS_STORE}debt_arc.xlsx"))
        xl = pd.ExcelFile(f"{XLS_STORE}debt_arc.xlsx")
        frame_result = pd.DataFrame(columns=["region", "total", "msp_total", "il_total"])
        for name in xl.sheet_names:
            frame = xl.parse(name, header=list(range(8)))
            frame = frame.iloc[:, :4]
            frame.columns = ["region", "total", "msp_total", "il_total"]
            frame["msp_total"] -= frame["il_total"]
            date_sheet = dt.datetime.strptime((name.split(' '))[-1], format("%d.%m.%Y"))
            frame['date_rep'] = date_sheet
            frame_result = pd.concat([frame_result, frame], axis=0, ignore_index=True)
        frame_result = frame_result[
            (frame_result['region'].str.contains('ОКРУГ') == False) & (frame_result['region'].str[0] != ' ')]
        frame_result = frame_result[(frame_result['region'].str.contains('округ') == False)]
        frame_result["okato_code"] = 0
        return frame_result