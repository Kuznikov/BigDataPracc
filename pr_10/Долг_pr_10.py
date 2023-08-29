#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import time

#zadanie 1
data = pd.read_csv('diabetes.csv')
print(data)
print(data.dtypes)
print(data.isna().sum()) # проверка на наличие пустых значений
data.duplicated() # проверка на дубликаты строк
data.drop_duplicates(inplace = True) # удаление дубликатов
data.duplicated() # проверка на дубликаты строк
print(data.describe())
plt.figure()
data.boxplot(column = 'Pregnancies')
plt.figure()
data.boxplot(column = 'DiabetesPedigreeFunction')


#zadanie 2

plt.figure(figsize = (8,6))
plt.hist(data['Outcome'], bins = 2, edgecolor = 'black', color = 'lightslategrey')

#zadanie 3

predictors = data.drop(columns = ['Outcome'],axis = 1) # в качестве предикторов используем все медицинские параметры
target = data.Outcome # в качестве целевой переменную используем наличие диабета

x_train, x_test, y_train, y_test = train_test_split(predictors, target, train_size = 0.8, # пропорции обучающей и тестовой выборок
                                                   shuffle = True, random_state = 128)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#zadanie 4

start_time = datetime.now() # замер времени выполнения алгоритма

model_LR = LogisticRegression(random_state = 128, max_iter = 790) # используем логистическую регрессию
model_LR.fit(x_train, y_train)
y_predict = model_LR.predict(x_test) # тесты на основе обученной модели
print(classification_report(y_predict, y_test))

print(datetime.now() - start_time) # замер времени выполнения алгоритма

fig = px.imshow(confusion_matrix(y_test, y_predict), text_auto = True)
fig.update_layout(xaxis_title = 'Target', yaxis_title = 'Prediction')

start_time = datetime.now() # замер времени выполнения алгоритма

param_kernel = ('linear', 'rbf', 'poly', 'sigmoid')
parameters = {'kernel': param_kernel}
model_SVC = SVC()
grid_search_svm = GridSearchCV(estimator = model_SVC, param_grid = parameters, cv = 6)
grid_search_svm.fit(x_train, y_train)

print(datetime.now() - start_time) # замер времени выполнения алгоритма
best_model = grid_search_svm.best_estimator_
best_model.kernel

svm_preds = best_model.predict(x_test)
print(classification_report(svm_preds, y_test))
fig = px.imshow(confusion_matrix(y_test, svm_preds), text_auto = True)
fig.update_layout(xaxis_title = 'Target', yaxis_title = 'Prediction')

start_time = datetime.now() # замер времени выполнения алгоритма

number_of_neighbors = np.arange(3, 10, 25)
model_KNN = KNeighborsClassifier()
params = {'n_neighbors': number_of_neighbors}

grid_search = GridSearchCV(estimator = model_KNN, param_grid = params, cv = 6)

print(datetime.now() - start_time) # замер времени выполнения алгоритма

grid_search.fit(x_train, y_train)
grid_search.best_score_ # лучшее значение macro-average

grid_search.best_estimator_
knn_preds = grid_search.predict(x_test)

print(classification_report(knn_preds, y_test))

fig = px.imshow(confusion_matrix(y_test, knn_preds), text_auto = True)
fig.update_layout(xaxis_title = 'Target', yaxis_title = 'Prediction')

#zadanie 5

print(classification_report(y_predict, y_test))
print(classification_report(svm_preds, y_test))
print(classification_report(knn_preds, y_test))
# %%
