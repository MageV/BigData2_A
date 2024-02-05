from clickhouse_sqlalchemy import engines
from sqlalchemy import Column, Integer, Date, String, Float

from config.appconfig import *


class AppTable(Base):
    __tablename__ = 'app_rows'
    __table_args__ = (
        engines.MergeTree(order_by=['id']),
        {'schema': 'app_storage'},
    )
    id = Column(Integer, primary_key=True)
    date_reg=Column(Date)
    workers=Column(Integer)
    okved=Column(String)
    region=Column(Integer)
    typeface=Column(Integer)
    key_pcs=Column(Float)
    usd_val=Column(Float)
    euro_val=Column(Float)
    chin_val=Column(Float)