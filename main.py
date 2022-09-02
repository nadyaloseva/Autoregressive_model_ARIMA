import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import tsa
import statsmodels.api as smapi
import statsmodels.api as sm
import prettytable




# считываем файл с данными
s_data = pd.read_csv('data.csv', sep=';', parse_dates=['REPORTDATE'], infer_datetime_format=True)
print(s_data.columns)
s_data.info()
print(s_data)

# простроение графика значений из dataseta
plt.plot(s_data['REPORTDATE'].values, s_data['VALUE'].values)

# месячные интервалы
s_data['REPORTDATE'] = pd.to_datetime(s_data['REPORTDATE'])
s_data = s_data.set_index('REPORTDATE')
weekly = s_data.resample('M').median().plot()
plt.show()

# вывод:график не имеет ярких выбросов и имеет ярко выраженный тренд роста.
# общая тенденция значения данных увеличивается с течением времени, все
# Среднее значение является функцией времени, а данные имеют тенденцию, поэтому они нестабильны.
# Из это можно сделать вывод о том, что ряд не является стационарным


# оценка стабильности данных временных рядов
def test_stationarity(timeseries):
# Здесь один год используется в качестве окна, значение каждого времени t заменяется
# средним значением за предыдущие 12 месяцев (включая его самого), а стандартное отклонение остается тем же.
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

# вывод графика за значениями :
    fig = plt.figure()
    fig.add_subplot()
    plt.plot(timeseries, color='blue', label='Исходные значения')
    plt.plot(rolmean, color='red', label='Скользящее среднее')
    plt.plot(rolstd, color='black', label='Стандартное отклонение')
    plt.legend(loc='best')
    plt.title('Скользящее среднее и стандартное отклонение ')
    plt.show()

# тест Дики-Фуллера:
    print('Результат теста Дики-Фуллера:')
    dftest = adfuller(timeseries, autolag='AIC')

# dftest - это значение обнаружения, значение p, количество задержек, количество использованных наблюдений
# и критическое значение для каждого уровня достоверности.
    dfoutput = pd.Series(dftest[0:4], index=['Значение обнаружения', 'p-value', 'Количество задержек', 'Количество используемых наблюдений'])
    for key, value in dftest[4].items():
        dfoutput['Уровень достровернсти для уровня(%s)' % key] = value
    print(dfoutput)
test_stationarity(s_data['VALUE'])

# Видно, что скользящее среднее / стандартное отклонение данных имеет тенденцию к увеличению и является нестабильным.
# И DF-test может четко указать, что данные нестабильны ни при какой степени достоверности.

# метод сглажевания
s_data_log = np.log(s_data)# логарифмизация
moving_avg = s_data_log.rolling(window=12).mean()
plt.plot(s_data_log, color='blue')
plt.plot(moving_avg, color='red')
plt.show()

s_data_log_moving_avg_diff = s_data_log-moving_avg
s_data_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(s_data_log_moving_avg_diff)

# из графика, что скользящее среднее / стандартное отклонение данных
# не имеет тенденцию к увеличению и является стабильным.
# проверим экспонентарный метод сглаживания

# Значение периода полураспада определяет коэффициент затухания alpha: alpha = 1-exp (log (0,5) / halflife)
expweighted_avg = s_data_log.ewm(halflife=12).mean()
s_data_log_ewma_diff = s_data_log - expweighted_avg
test_stationarity(s_data_log_ewma_diff)

s_data_log_diff = s_data_log - s_data_log.shift()
s_data_log_diff.dropna(inplace=True)
test_stationarity(s_data_log_diff)

def decompose(timeseries):
    # Возврат состоит из трех частей: тренд (часть тренда), сезонная (сезонная часть) и остаток (остаточная часть)
    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(s_data_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return trend, seasonal, residual

decompose(s_data_log)

# из графиков видно, что данные имеют тренд, однако сезонность отсутстует
# после удаления тренда можно будет сказать, что данные имеют стабильность исключая выпадающие значения в 2014,2015 годах


# После удаления тренда и сезонности только остаточная часть обрабатывается как данные желаемого временного ряда
trend, seasonal, residual = decompose(s_data_log)
residual.dropna(inplace=True)
test_stationarity(residual)

# ACF and PACF:
lag_acf = acf(s_data_log_diff, nlags=20)
lag_pacf = pacf(s_data_log_diff, nlags=20, method='ols')


# вывод ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(s_data_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(s_data_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.show()

# вывод PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96 / np.sqrt(len(s_data_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96 / np.sqrt(len(s_data_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

#модель ARIMA
date_index2 = pd.date_range('2013-12-30', '2019-10-01', freq='M')
arima = smapi.tsa.arima.ARIMA(s_data.reindex(date_index2), order=(1, 1, 1), freq='M')
result = arima.fit()
print(result.aic, result.bic, result.hqic)

plt.plot(s_data_log_diff)
plt.plot(result.fittedvalues, color='red')
plt.title('ARIMA RSS: %.4f' % sum(result.fittedvalues - s_data_log_diff['VALUE']) ** 2)
plt.show()

# Модель прогнозирования
pred = result.predict('2019-11-01', '2020-10-01', typ='levels')
print(pred)
x = pd.date_range('2020-01-01', '2020-12-01')
plt.plot(x[:70 ], s_data.reindex(date_index2)['VALUE'])
# lenth = len()
plt.plot(pred)
plt.show()
print('end')

