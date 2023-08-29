import random
import pandas as pd
import numpy as np
import stats as st
import scipy.stats as sts
import plotly
import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns


def zadanie():
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    # zadanie_1
    insurance = pd.read_csv('C:\\Users\\Maksim\\Desktop\\learning\\7_semestr\\bigData\\pr_4\\insurance.csv')
    # zadanie_2
    print(insurance.describe())
    # zadanie_3
    insurance.hist(color='green', edgecolor='black')
    plt.show()
    # zadanie_4
    # вычисляем меры центральной тенденции bmi
    mode_bmi = sts.mode(insurance['bmi'])  # мода
    med_bmi = np.median(insurance['bmi'])  # медиана
    mean_bmi = np.mean(insurance['bmi'])  # среднее
    print('Мода bmi: ', mode_bmi,
          '\nМедиана bmi: ', med_bmi,
          '\nСреднее bmi: ', round(mean_bmi, 1))
    # вычисляем меры разброса bmi
    scope_bmi = insurance['bmi'].max() - insurance['bmi'].min()  # размах
    std_bmi = insurance['bmi'].std()  # стандартное отклонение
    iqr_bmi = sts.iqr(insurance['bmi'], interpolation='midpoint')  # межквартильный размах
    print('Размах bmi: ', scope_bmi,
          '\nСтандартное отклонение bmi: ', round(std_bmi, 2),
          '\nМежквартильный размах bmi: ', round(iqr_bmi, 2))

    # вычисляем меры центральной тенденции charges
    mode_charges = sts.mode(insurance['charges'])
    med_charges = np.median(insurance['charges'])
    mean_charges = np.mean(insurance['charges'])
    print('\nМода charges: ', mode_charges,
          '\nМедиана charges: ', med_charges,
          '\nСреднее charges: ', round(mean_charges, 1))
    # вычисляем меры разброса charges
    scope_charges = insurance['charges'].max() - insurance['charges'].min()  # размах
    std_charges = insurance['charges'].std()  # стандартное отклонение
    iqr_charges = sts.iqr(insurance['charges'], interpolation='midpoint')  # межквартильный размах
    print('Размах charges: ', round(scope_charges, 2),
          '\nСтандартное отклонение charges: ', round(std_charges, 2),
          '\nМежквартильный размах charges: ', round(iqr_charges, 2))



    name_central_trend = ['Мода', 'Медиана', 'Среднее']  # названия мер центральной тенденции
    name_scatter = ['Размах', 'Станд. откл.', 'Межквартильный размах']  # названия мер разброса

    central_trend_bmi = [mode_bmi.mode, med_bmi, mean_bmi]  # список мер центральной тенденции bmi
    central_trend_charges = [mode_charges.mode, med_charges, mean_charges]  # список мер центральной тенденции charges

    scatter_bmi = [scope_bmi, std_bmi, iqr_bmi]  # меры разброса bmi
    scatter_charges = [scope_charges, std_charges, iqr_charges]  # меры разброса charges

    fig, ax = plt.subplots(2, 2, figsize=(14, 14))  # создание нескольких графиков
    ax[0][0].bar(name_central_trend, central_trend_bmi, edgecolor='black', linewidth=2, width=0.5)
    ax[0][0].grid(alpha=0.4)  # сетка
    ax[0][0].set_title('Меры центральной тенденции bmi', size=16)

    ax[0][1].bar(name_central_trend, central_trend_charges, edgecolor='black', color='gold', linewidth=2, width=0.5)
    ax[0][1].grid(alpha=0.4)
    ax[0][1].set_title('Меры центральной тенденции charges', size=16)

    ax[1][0].bar(name_scatter, scatter_bmi, edgecolor='black', color='green', linewidth=2, width=0.5)
    ax[1][0].grid(alpha=0.4)
    ax[1][0].set_title('Меры разброса bmi', size=16)

    ax[1][1].bar(name_scatter, scatter_charges, edgecolor='black', color='red', linewidth=2, width=0.5)
    ax[1][1].grid(alpha=0.4)
    ax[1][1].set_title('Меры разброса charges', size=16)
    plt.show()

    # zadanie_5
    plt.figure(figsize=(8, 8))
    plt.boxplot([insurance['charges']], labels=['Charges'], vert=False)
    plt.show()

    plt.boxplot([insurance['age'], insurance['bmi'], insurance['children']],
                labels=['Age', 'BMI', 'Children'], vert=False)
    plt.show()

    # zadanie_6
    n = 200
    samples_number = 300
    t = tuple(insurance['bmi'])
    mean_list = []

    # создаем samples_number раз выборку размера n, находим в ней среднее и добавляем его в список средних
    for i in range(samples_number):
        data_sample = random.sample(t, n)
        mean_list.append(np.mean(data_sample))

    sns.displot(mean_list, kde = False) # вывод гистограммы
    plt.show()

    print('Среднее распределения: ', np.mean(mean_list))
    print('Стандартное отклонение распределения: ', st.pstdev(mean_list))

    # zadanie_7-8
    # вычисляем 95% и 99% доверительные интервалы для charges
    charges_conf_int95 = sts.norm.interval(0.95, loc=mean_charges, scale=std_charges)
    charges_conf_int99 = sts.norm.interval(0.99, loc=mean_charges, scale=std_charges)
    print('95% доверительный интервал для charges: ', charges_conf_int95)
    print('99% Доверительный интервал для charges: ', charges_conf_int99)

    # вычисляем 95% и 99% доверительные интервалы для bmi
    bmi_conf_int95 = sts.norm.interval(0.95, loc=mean_bmi, scale=std_bmi)
    bmi_conf_int99 = sts.norm.interval(0.99, loc=mean_bmi, scale=std_bmi)
    print('\n95% доверительный интервал для bmi: ', bmi_conf_int95)
    print('99% доверительный интервал для bmi: ', bmi_conf_int99)

    # делаем две выборки ИМТ из 300 значений
    bmi_sample1 = insurance['bmi'].sample(300)
    bmi_sample2 = insurance['bmi'].sample(300)
    print('KS-тест ИМТ: ', sts.kstest(bmi_sample1, bmi_sample2))  # выполняем KS-тест для этих выборок

    # делаем две выборки расходов из 300 значений
    charges_sample1 = insurance['charges'].sample(300)
    charges_sample2 = insurance['charges'].sample(300)
    print('KS-тест расходов: ', sts.kstest(charges_sample1, charges_sample2))  # выполняем KS-тест для этих выборок

    # создаем q-q plot
    qq_plot = sns.jointplot(
        # создаем нормальное распределение размером с генеральную совокупность и указываем по оси X
        x=np.random.normal(size=1338),
        y=insurance['bmi'],  # указываем по оси Y ИМТ
        kind='reg',  # тип графика
        color='chocolate',  # цвет графика
        line_kws={'lw': 1, 'color': 'black'}  # цвет линии

    )

    # создаем q-q plot
    qq_plot_charges = sns.jointplot(
        # создаем нормальное распределение размером с генеральную совокупность и указываем по оси X
        x=np.random.normal(size=1338),
        y=insurance['charges'],  # указываем по оси Y расходы
        kind='reg',  # тип графика
        color='midnightblue',  # цвет графика
        line_kws={'lw': 1, 'color': 'black'})  # цвет линии


if __name__ == '__main__':
    zadanie()
