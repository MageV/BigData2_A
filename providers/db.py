from config.appconfig import *


def prepare_tables(table):
    if table == 'cbr':
        click_client.command("alter table app_cbr delete where 1=1")
    if table == 'app':
        click_client.command("alter table app_row delete where 1=1")


def prepare_ml_data():
    pass