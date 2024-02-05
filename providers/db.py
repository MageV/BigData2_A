from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.appconfig import *


class ClickHouseProvider:
    def __init__(self):
        self.__engine = create_engine(conn_str)
        self.__session = sessionmaker(bind=self.__engine)()

    def db_append(self, rowlist):
        with self.__session.begin() as session:
            for item in rowlist:
                session.add(item)
            session.commit()
