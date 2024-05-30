# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:21:48 2023

@author: Danny Ortega
"""

import os
import sys
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from shapely.geometry import Point
import re
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


db = pd.read_excel('datos.xlsx')

datos = db[['GASODUCTOS','PORCENTAJE_CORROSION', 'ALTURA'
              ,'CO2', 'H2S', 'H20', 'HUMEDAD',
               'TEMPERATURA_GAS',
              'PRESION'
]]

#######################
label=list(map(lambda x: 1 if x>=40 else 0, datos.PORCENTAJE_CORROSION))

datos['label']=label


X=datos.drop(['label','GASODUCTOS'],axis=1)

X_pred =X.drop('PORCENTAJE_CORROSION', axis=1)

y=datos['label']

#### GRAFICO DE CORRELACION #######

correlacion = X.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de calor de correlación')
plt.show()


###### SUBMUESTREO ##########
rus = RandomUnderSampler(random_state=42,sampling_strategy=0.6)

X_res, y_res = rus.fit_resample(X,y)

print('Resampled dataset shape %s' % Counter(y_res))

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0,stratify=y_res)

################
""" DEBIDO A QUE ES NECESARIO TENER VALORES BINARIOS PARA HACER UN SUBMUESTREO, SE CREO UNA COLUMNA LABEL DONDE SE MARCA LOS REGISTROS
QUE TENGAN UN PORCENTAJE  MAYOR O IGUAL A 40 CON "1" Y LOS QUE NO CON "0" Y LUEGO REEMPLAZAMOS ESTOS LABELS POR SU RESPECTIVA PORCENTAJE
"""

y_train = pd.DataFrame(y_train)
y_train['PORCENTAJE_CORROSION'] = X_train['PORCENTAJE_CORROSION']
y_train= y_train.drop(['label'], axis=1)
y_train = y_train.dropna()

y_test = pd.DataFrame(y_test)
y_test['PORCENTAJE_CORROSION'] = X_test['PORCENTAJE_CORROSION']
y_test = y_test.drop(['label'], axis=1)
y_test = y_test.dropna()

X_train = X_train.drop(['PORCENTAJE_CORROSION'], axis=1)
X_test = X_test.drop(['PORCENTAJE_CORROSION'], axis=1)

tramo2 = X_test
tramo2.reset_index(drop = True, inplace = True)

###### CREACION Y ENTRENAMIENTO DE MODELOS ########
GBR = GradientBoostingRegressor(random_state=42)
GBR.fit(X_train, y_train)

######### PREDICCION ###########
y_pred = GBR.predict(X_test)
y_pred = pd.DataFrame({'y_pred': y_pred})


####### EVALUACION DEL MODELO, GRAFICOS Y CONCATENACION DE LOS DATOS

##### IMPORTANCIA DE VARIABLES ########

feat_dict= {}
for col, val in sorted(zip(X_train.columns, GBR.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val

feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

#### SCORE ######

score1=r2_score(y_test,y_pred)


####### VISUALIZACION DE LOS PORCENTAJES Y LAS PREDICCIONES #######

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
overlapping = 0.150
line1 = plt.plot(y_pred, c='red', alpha=overlapping, lw=1)
line2 = plt.plot(np.array(y_test), c='green', alpha=overlapping,lw=1)
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.scatter(X_test['ALTURA'], y_test, color='blue', label='Datos reales')  # Datos originales
ax.scatter(X_test['ALTURA'], y_pred, color='red', label='Predicciones del modelo GBR')  # Predicciones del modelo GBR
ax.set_xlabel('Variable independiente 1 (X1)')
ax.set_ylabel('Variable independiente 2 (X2)')
plt.title('Gráfico 3D de predicciones de modelo GBR')
plt.legend()
plt.show()

###### UNION DE DATOS ########
pred_GBR = pd.DataFrame(y_pred)
pred_GBR.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

verificar_GBR = pd.concat([tramo2, y_test, pred_GBR], axis = 1)

X=datos.drop(['label','GASODUCTOS', 'PORCENTAJE_CORROSION'],axis=1)

pred_total = GBR.predict(X)

db['PREDICCION'] = pred_total

