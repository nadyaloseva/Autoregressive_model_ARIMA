import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt


s_data = pd.read_csv('data.csv', sep=';') #считываем файл с данными
#print(s_data) #проверка правильности введенных данных
print(s_data.columns) #выводим название столбцов для дальнейшей работы с ними
print(s_data.dtypes) #проверяем типы столбцов
s_data.astype({'VALUE': 'int'}).dtypes #меняем формат столбца с числовыми значениями
s_data.dtypes.value_counts()
result, timezones = array_strptime(arg, fmt, exact=exact, errors=errors)