# traning_sber

После анализа набора данных, которые мы получили для обработки, мы выяснили, что он состоит из 2111 строк и 2 стоблцов с значениями даты и 
стабильной части полученной суммы. Для наглядности полученных данных построем график. 
График не имеет ярких выбросов и имеет ярко выраженный тренд роста общая тенденция значения данных увеличивается с течением времени, все. 
Среднее значение является функцией времени, а данные имеют тенденцию, поэтому они нестабильны. Из это можно сделать вывод о том, что ряд 
не является стационарным. 
Далее проведем тест стабильности данных, с помощью функции test_stationarity. По скользящей средней видно, что временной ряд определенный тренд 
возрастающего характера. Стандратное отклонение показывает, что количество выбросов не велико, из чего можно сделать вывод, о том, что имеет сезонный характер.

Так же сделает тест Дики-Фуллера: 
Значение обнаружения                       -0.228894
p-value                                     0.934989
Количество задержек                        26.000000
Количество используемых наблюдений       2084.000000
Уровень достровернсти для уровня(1%)       -3.433492
Уровень достровернсти для уровня(5%)       -2.862928
Уровень достровернсти для уровня(10%)      -2.567509

Полученные данные так подверждают тренд с возрастающего характера. Поэтому временной ряд нельзя назвать стационарным.
Для того, что исправить используем метод сглажевания, после которого скользящее среднее / стандартное отклонение данных не имеет тенденцию к увеличению
и является стабильным.

Результат теста Дики-Фуллера:
Значение обнаружения                    -8.882709e+00
p-value                                  1.304850e-14
Количество задержек                      2.600000e+01
Количество используемых наблюдений       2.073000e+03
Уровень достровернсти для уровня(1%)    -3.433508e+00
Уровень достровернсти для уровня(5%)    -2.862935e+00
Уровень достровернсти для уровня(10%)   -2.567513e+00

Для более точного результата проверим экспонентарный метод сглаживания. Однако, этот метод не улучшил, полученным в прошлом результат. Поэтому остается
только удалить тренд для использования метода ARMI, так как сезонность отсутвует. 

Результат теста Дики-Фуллера:
Значение обнаружения                    -1.801539e+01
p-value                                  2.706074e-30
Количество задержек                      2.600000e+01
Количество используемых наблюдений       2.078000e+03
Уровень достровернсти для уровня(1%)    -3.433501e+00
Уровень достровернсти для уровня(5%)    -2.862932e+00
Уровень достровернсти для уровня(10%)   -2.567511e+00

После проверки теста Дики-Фуллера видно, что стоит использовать функция автокорреляции ошибок ARIMA(2,1,2). После чего получаем значение стабильная часть
на ближайшие 12 месяцев. 

2019-11-30    6.396678e+10
2019-12-31    6.477434e+10
2020-01-31    6.557601e+10
2020-02-29    6.637716e+10
2020-03-31    6.717823e+10
2020-04-30    6.797924e+10
2020-05-31    6.878020e+10
2020-06-30    6.958111e+10
2020-07-31    7.038197e+10
2020-08-31    7.118277e+10
2020-09-30    7.198353e+10
2020-10-31    7.278424e+10



 
