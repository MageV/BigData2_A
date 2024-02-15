# Пакет построения ML моделей #

# AI_HYPER #

Файл параметров для построения моделей

# AI_MODEL #
*ai_learn(mean_over, features=None, scaler=AI_SCALER.AI_NONE,<br> 
models_class=AI_MODELS.AI_REGRESSORS,<br> msp_class=MSP_CLASS.MSP_UL)*<br>
Основная функция для расчета ML моделей. Входные данные<br>
__mean_over__ - ключ использования зависимых переменных.<br>
__features__ - Перечень полей DataFrame,используемых для построения модели.<br>
__scaler__ - вид Scaler функции для входных переменных <br>
__models_class__ - классификатор семейства моделей для расчета<br>
__msp_class__ - вид расчета: по микропредприятиям или ИП<br>
