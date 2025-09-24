# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:44:27 2024

@author: FernandezD3
"""
#++++++    ESTE CODIGO A SIDO DISEÑADO EN TIPO DEMOSTRACIÓN PARA COMPARTIRLO EN EL TFM +++++++++++++


#cargar librerias
reset -f
#import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import keras.utils
from keras import utils as np_utils
import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys
import seaborn as sns



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
from pandas import Series



from pandas import Series
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from operator import itemgetter




from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import sklearn.metrics 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Activation, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.backend import expand_dims
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
  
from sklearn.model_selection import train_test_split






        

#%% 

# ++++++++++ Proceso 19  +++++++++
# CREAR ARRAYS 3D


import keras
import keras.utils
from keras import utils as np_utils
import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

from pandas import Series

import os


os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\pruebaspython\test enrtegar")


BBDDaceitesmodelo = pd.read_excel(io = "testdatosmodelo3estandtotalaceitecodificado.xlsx", sheet_name="Sheet1")
test=BBDDaceitesmodelo.copy()

        
#crear y rellenar el array 3D de test

listawtgrevisados=[]
eje0=len(test['reportWTGbis'].unique())
test['Yrun'] = test['Yrun'].astype(int)     
test['Yex'] = test['Yex'].astype(int) 
x_test=np.zeros(shape=(eje0, 35, 9))

y_test=np.zeros(shape=(eje0,2))


j=0
for i in test['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        
    
        muestratest=test.loc[test['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        muestratestarray=np.array(muestratest.loc[:, ['PC_4esRob', 'PC_6esRob', 'PC_14esRob','Designrevision_0','Designrevision_1','Designrevision_2','Designrevision_3','Designrevision_4','Designrevision_5']])
        #muestratestarray=np.array(muestratest.loc[:, ['PC_4esRob', 'PC_6esRob', 'PC_14esRob']])
        x_test[j,:,:] = muestratestarray
        
        
        
        
        muestratest=muestratest.reset_index()
        muestratest=muestratest.drop('index', axis=1)# se modifica index
        y_test[j,:]=np.array(muestratest.loc[34, ['Yrun', 'Yex']])
        
        
    
        j=j+1        
        
        
#%% 




###llamo al modelo


os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\pruebaspython\test enrtegar")

fnameM = 'C:/Users/FernandezD3/OneDrive - Nordex SE/Documentos/Master thesis/Master/Master/pruebaspython/test enrtegar/checkpoint.09-0.73.keras'
M=keras.models.load_model(fnameM)

predictions = M.predict(x_test,batch_size=500)


# summary del modelo
print(M.summary())


os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\pruebaspython\test enrtegar")



####visualizo resultados


print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1))) # las metricas se calculan con los resultados de test




y_testconvconf=y_test[:,1]

predictionsconvconf=np.where(predictions >= 0.5, 1, 0)
predictionsconvconf=predictionsconvconf[:,1]

names=["Running","Exhanged"]


#condusion matrix normalizada
cm=confusion_matrix(y_testconvconf,predictionsconvconf)
cmn = cm.astype('float')
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=names, yticklabels=names)# a veces no funciona y no se porque
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion matrixtest.png",dpi=1000.0,bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
#reporte métricas de clasificacion
reportPCs = classification_report(y_testconvconf,predictionsconvconf)

sklearn.metrics.accuracy_score(y_testconvconf,predictionsconvconf)
sklearn.metrics.precision_score(y_testconvconf,predictionsconvconf)
sklearn.metrics.recall_score(y_testconvconf,predictionsconvconf)




#%%

#codigo para crear la matriz de riesgo. Es la matriz con los resultados finales predichos por el modelo
listawtgrevisados=[]
WTG='' 
j=0
risk=''

matrizrisk= pd.DataFrame(columns=['wtg', 'riskfactor'],index=range(292))
datosmodelo1resmatriz=test.copy()
for i in datosmodelo1resmatriz['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        listawtgrevisados.append(wtg)
        muestra=datosmodelo1resmatriz.loc[datosmodelo1resmatriz['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        aero=muestra['reportWTGbis'].tail()
        aero1=muestra.loc[muestra.index[-1], "reportWTGbis"]
        predictions1=pd.DataFrame(predictions)  
        prediccion=predictions1.iloc[j,1]
      
        
        if prediccion < 0.5: 
            risk="low"
        elif prediccion >=0.5 and prediccion<0.7: 
            risk="middle"
        elif prediccion>=0.7: 
            risk="high"
       
        matrizrisk.iloc[j,0]=aero1
        matrizrisk.iloc[j,1]=risk
        j=j+1

            

matrizrisk.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\pruebaspython\test enrtegar\matrizriesgofinal.xlsx', index = False)
     


#+++++++++++++++   FIN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


   