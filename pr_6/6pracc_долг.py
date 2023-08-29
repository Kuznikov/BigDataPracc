import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import random

data = pd.read_csv('ECDCCases.csv') # чтение csv файла
print (data)
print(data.isna().sum()) # проверка на наличие пустых значений

# вывод кол-ва пропущенных значений в процентах
for column in data.columns:
    missing = np.mean(data[column].isna()*100)
    print(f" {column} : {round(missing, 1)}%")

# удаление двух признаков с наибольшим кол-вом пропусков
data.drop('Cumulative_number_for_14_days_of_COVID-19_cases_per_100000', axis = 1, inplace = True)
data.drop('geoId', axis = 1, inplace = True)
print(data.isna().sum()) #проверка на наличие пустых значений

data.countryterritoryCode.fillna(value = 'other', inplace = True) # заполнение пустых значений
data.countryterritoryCode.unique() # проверка уникальных значений

med_popData2019 = data.popData2019.median() # вычисление медианы
data.popData2019.fillna(med_popData2019, inplace = True) # замена пустых значений медианным

print(data.isna().sum()) #проверка на наличие пустых значений

data.describe()

plt.figure(figsize = (3,3))
data.boxplot(column = 'cases')

plt.figure(figsize = (3,3))
data.boxplot(column = 'popData2019')

ddays = 0 # кол-во дней с >3000 смертей
countries = set()

for row in data.itertuples(): # проход по строкам датафрейма
    if row.deaths > 3000:
        ddays += 1
        countries.add(row.countriesAndTerritories)

print('Количество дней с более 3000 смертей: ', ddays)
print(countries)

data.duplicated() # проверка на дубликаты строк

data.drop_duplicates(inplace = True) # удаление дубликатов
data.duplicated() # проверка на дубликаты строк

data_bmi = pd.read_csv('bmi.csv')
print(data_bmi)

data_bmi.region.unique()

n = 100 # размер выборок

# разделенные генеральные совокупности
gp_north = list()
gp_south = list()

for row in data_bmi.itertuples():
    if row.region == 'northwest':
        gp_north.append(row.bmi)
    else:
        gp_south.append(row.bmi)
        
# взятие выборок        
sample_north = random.sample(gp_north, n)
sample_south = random.sample(gp_south, n)

res1 = st.shapiro(sample_north)
res2 = st.shapiro(sample_south)
print(res1)
print(res2)

res = st.bartlett(sample_north, sample_south)
print(res)

t_res = st.ttest_ind(sample_north, sample_south)
print(t_res)

# создание датафрейма с полученными и ожидаемыми значениями
throws = pd.DataFrame(np.array([[97, 100], [98, 100], [109, 100], [95, 100], [97, 100], [104, 100]]),
                     columns = ['observed', 'expected'])
print(throws)

st.chisquare(throws['observed'], throws['expected'])

data = pd.DataFrame({'Женат': [89,17,11,43,22,1],
                     'Гражданский брак': [80,22,20,35,6,4],
                     'Не состоит в отношениях': [35,44,35,6,8,22]})
data.index = ['Полный рабочий день','Частичная занятость','Временно не работает',
              'На домохозяйстве','На пенсии','Учёба']
print(data)

st.chi2_contingency(data)[:3] # тестовая статистика, p-значение, степени свободы
