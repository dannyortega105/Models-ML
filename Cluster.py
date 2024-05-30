# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 08:17:52 2023

@author: Danny Ortega
"""

import database_models
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans


db = database_models.ili()

k_datos = db[['GASODUCTOS','PORCENTAJE_CORROSION','LARGO',  'ANCHO','ALTURA']]
    
k_datos.rename(columns={'ALTURA': 'ALTURA_x'}, inplace=True)
    
    
############################### CREACION DE MODELO ##############################
    
x_train=k_datos.copy()
x_train=x_train.drop(['GASODUCTOS'],axis=1)

####### ESTANDARIZAR LOS DATOS ###########3        
for c in x_train.columns:
    pt = PowerTransformer()
    x_train.loc[:, c] = pt.fit_transform(np.array(x_train[c]).reshape(-1, 1))
        
kmeans = KMeans(init="k-means++", n_clusters=5, n_init=100, random_state=42)
kmeans.fit(x_train)
test_isolation=kmeans.predict(x_train)
test_isolation=pd.DataFrame(test_isolation)
test_isolation.reset_index(drop=True,inplace=True)
verificar_iso=pd.concat([k_datos,test_isolation],axis=1)
verificar_iso=verificar_iso.drop(['GASODUCTOS'], axis=1)
verificar_iso['Segmentos']=verificar_iso[0]
verificar_iso=verificar_iso.drop(0, axis=1)
k_datos['Segmentos'] = verificar_iso['Segmentos']


#### INTERPRETRACION ####
    
def interpretacion(datos, numeric, categoric=None):
    # porcentaje en cada cluster
    ax = sns.countplot(y='Anomalia', data=datos)
    plt.title('Porcentaje Clusters')
    plt.xlabel('%')
    total = len(datos['Anomalia'])
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

    # boxplots
    for col in numeric:
        fig, axs = plt.subplots(1, 2, figsize=(25, 9.6))
        # Agregamos titulo a la figura.
        fig.suptitle(col.upper().replace("_", " "), fontsize=20)
        grouped_data = datos[[col, 'Anomalia']].groupby('Anomalia')
        # Estadísticas para todas las columnas numéricas por cluster
        tab = np.round(grouped_data.describe(), 2)
        the_table = table(axs[1], tab, loc='center', cellLoc='center')
        the_table.set_fontsize(20)
        the_table.scale(1.3, 2)
        axs[1].axis("off")

        plt.axes(axs[0])
        ax = sns.boxplot(y=col, x='Anomalia', data=datos, orient="v", width=0.45, palette="Set1")

        # Pesonalizamos la gráfica
        ax.set_ylabel(col.replace("_", " "), fontsize=15)
        ax.set_xlabel("Número Clusters", fontsize=15)

datos_iso = k_datos.copy()
datos_iso = datos_iso[datos_iso['ENGROUTEID'] == 'T_BAL_HAT']
datos_iso['Anomalia'] = datos_iso[0]
datos_iso = datos_iso.drop(['ENGROUTEID',  0],axis=1)
datos_modelado = datos_iso.drop(['Anomalia'],axis=1)

interpretacion(datos_iso, datos_modelado)