# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:20:15 2023

@author: Danny Ortega
"""
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
import database_models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from collections import Counter
import lightgbm as ltb
import shap
from sklearn.model_selection import cross_val_score
from lightgbm import plot_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

########### IMPORTE DE LA INFORMACION ###########
db = database_models.data()

######## SE SELECCIONAN LAS VARIABLES QUE SE UTILIZARAN
k_datos = db[['GASODUCTOS','PORCENTAJE_CORROSION', 'ALTURA'
              ,'CO2', 'H2S', 'H20', 'HUMEDAD',
               'TEMPERATURA_GAS',
              'PRESION'
]]

""" DEBIDO A QUE ES NECESARIO TENER VALORES BINARIOS PARA LA CLASIFICACION Y PARA HACER UN SUBMUESTREO, SE CREO UNA COLUMNA LABEL DONDE SE MARCA LOS REGISTROS
QUE TENGAN UN PORCENTAJE  MAYOR O IGUAL A 40 CON "1" Y LOS QUE NO CON "0" Y LUEGO REEMPLAZAMOS ESTOS LABELS POR SU RESPECTIVA PORCENTAJE
"""

label=list(map(lambda x: 1 if x>=40 else 0, k_datos.DEPTHMAXPERC))

k_datos['label']=label

X=k_datos.drop(['label','GASODUCTOS','PORCENTAJE_CORROSION'],axis=1)

y=k_datos['label']

########### SUBMUESTREO #########
rus = RandomUnderSampler(random_state=42,sampling_strategy=0.3)

X_res, y_res = rus.fit_resample(X,y)

print('Resampled dataset shape %s' % Counter(y_res))

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0,stratify=y_res)


######### SOBREMUESTREO #######
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))


############ ENTRENAMIENTO DEL MODELO #########

clf_km = ltb.LGBMClassifier(random_state=123)

clf_km.fit(X_res, y_res)


#### PREDICCION #####

y_pred=clf_km.predict(X_test)

f1_score(y_test, y_pred,average='micro')


######## EVALUACION DEL MODELO ########
clf_km.score(X_test, y_test)

cv_scores_km = cross_val_score(
    clf_km, X_res, y_res, scoring='f1_weighted')
print(f'CV F1 score is {np.mean(cv_scores_km)}')

 
# Características importantes de forma gráfica

explainer_km = shap.TreeExplainer(clf_km)
shap_values_km = explainer_km.shap_values(X_res)
shap.summary_plot(shap_values_km, X_res,
                  plot_type="bar", plot_size=(15, 10))


plot_importance(clf_km)
plot_importance(clf_km, importance_type="gain")
#, importance_type="gain"
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True);

cm = confusion_matrix(y_test, y_pred, labels=clf_km.classes_)
cm

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


##### RESULTADO DE LOS DATOS ######
xT_prueba = X_test.copy()
xT_prueba.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
y_T = pd.DataFrame(y_test)
y_P = pd.DataFrame(y_pred)

xT_prueba['test'] = y_T['label']
xT_prueba['pred'] = y_P[0]


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



