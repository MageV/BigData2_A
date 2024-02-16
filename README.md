# Проект анализ МСП

## Исходные данные ##

Cайт ФНС в виде zip архива размером 2 gb( внутри большое количество xml).
Динамика изменений архива 1 раз месяц. Есть архив файлов.
Сайт cbr.ru - данные по ключевой ставке и курсам валют

## Обработка ##
Производится дополнительная догрузка данных  с сайта cbr.ru<br>
 в части ключевой ставки и курсов валют. Исходные данные (ФНС) очищаются и пребразуются<br>
к форме, удобной для решения в рамках задачи классификации. 

## Выходные данные ##
Файл ML - модели.

# Результаты построения моделей #
__Model:'RandomForestClassifier'__<br>
*R1: 0.6731148128246462<br>*
__Model:'DecisionTreeClassifier'__<br>
*R1: 0.6731148128246462<br>*
__Model:'ExtraTreesClassifier'__<br>
*R1: 0.6731148128246462*

# TO-DO #
Необходимо повысить качество модели путем использования открытых данных форм банковской <br>
отчетности из источников cbr.ru для top-50 банков в части кредитной массы для ЮЛ и ИП

