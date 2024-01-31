import asyncio

import pandas as pd

from apputils.log import write_log
from config.appconfig import *
import datetime as dt


class ResultStorage:

    def __init__(self):
        self.__results = list()

    def append(self, item):
        if len(self.__results) <= 100000:
            self.__results += item


    def dump(self, queue):
        asyncio.run(write_log(message=f"write file to disk.started at:{dt.datetime.now()}", severity=SEVERITY.INFO))
        try:
            df = pd.DataFrame(self.__results)
            name = f"{RESULT_STORE}temp_{self.__counter}.csv"
            df.to_csv(name, chunksize=25000, header=False, index=False)
        except Exception as ex:
            asyncio.run(write_log(message=f"write file to disk failed:{ex.__str__()}", severity=SEVERITY.ERROR))
        self.__results.clear()
