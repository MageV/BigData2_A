# Пакет поставщиков данных #

## WEB ##
### модуль для работы с Internet ресурсами ###
__WebScraper()__ - класс для работы с источниками данных Internet<br>

__WebScraper.get_FNS()__ - функция получения данных из ФНС. Результат - файл архива<br>
в каталоге накопления архивов <br>
__url__ - адрес для скачивания архива ФНС
<br>
__WebScraper.get_rates_cbr()__ - __deprecated__

__WebScraper.get_regions()__ - возвращает DataFrame в виде ОКАТО(Код),ОКАТО(Регион) <br>

__WebScraper.get_sors_archive()__ -  функция загрузки архивов по кредитам с сайта Банка России<br>

__WebScraper.get_sors(processors)__ - функция загрузки актуальной информации по кредитам<br> с сайта Банка России<br>


_WebScraper().get_data_from_cbr(mindate,maxdate)_ - функция получения данных с сайта Банка России<br>
Запрашиваются данные по ключевой ставке, курсу доллара и евро. Результат DataFrame

## DB_CLICKHOUSE ##
### Модуль для работы с БД(ClickHouse) ##
_db_prepare_tables(table)_ - очистка таблиц ClickHouse перед загрузкой данных<br>

_db_get_frames_by_facetype(ft, mean_over)_ получение DataFrame ClickHouse в различных разрезах<br>

_db_ret_okato()_ получение справочника ОКАТО из предварительно загруженных данных ClickHouse<br>

_db_recode_workers(df)_ - Служебная функция.<br>
Перекодирование количества работающих в разрезе регионов.<br>
Перекодирование выполняется в бинарной форме по отношению к предыдущему периоду:<br>
__(less-or-equal) - 0__ <br>
__(greater) - 1__ <br>

## DF ##
_df_clean_for_ai(df: pd.DataFrame)_ - основная функция очистки данных

## DB_HIVE ##
### Модуль для работы с БД(Hive) ##
Функционал аналогичен модулю __DB_CLICKHOUSE__