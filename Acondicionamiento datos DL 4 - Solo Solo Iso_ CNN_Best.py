# -*- coding: utf-8 -*-

"""
Created on Fri Aug  9 21:07:43 2024

@author: FernandezD3
"""



#++++++++++++++codigo para poner en tablamodelo, los valores de "GBXs status" en one hot encoding
reset -f

#cargar librerias
#import tensorflow as tf


import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

from pandas import Series

# Establecemos el directorio de trabajo


#%%

        

#%% 


# ++++++++++ Proceso 19  +++++++++
# CREAR ARRAYS 3D

reset -f
import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

from pandas import Series

import os


os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\Modelos predictivos\Cod. acondicionamiento y tablas utilizadas")
BBDDaceitesmodelo = pd.read_excel(io = "traindatosmodelo3estandtotalaceite.xlsx", sheet_name="Sheet1")
train=BBDDaceitesmodelo.copy()

BBDDaceitesmodelo = pd.read_excel(io = "testdatosmodelo3estandtotalaceite.xlsx", sheet_name="Sheet1")
test=BBDDaceitesmodelo.copy()





listawtgrevisados=[]
eje0=len(train['reportWTGbis'].unique())
train['Yrun'] = train['Yrun'].astype(int)     
train['Yex'] = train['Yex'].astype(int) 
x_train=np.zeros(shape=(eje0, 35,9))#crea un aray de train de 3D de (len(train['reportWTGbis'].unique()) x 35 x 9) dimensiones
y_train=np.zeros(shape=(eje0,2))#crea un aray de test de train de 2D
j=0
for i in train['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        
    
        muestratrain=train.loc[train['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        
        #convierte la muestra en una array de 2D
        muestratrainarray=np.array(muestratrain.loc[:, ['PC_4esRob', 'PC_6esRob', 'PC_14esRob','Designrevision_0','Designrevision_1','Designrevision_2','Designrevision_3','Designrevision_4','Designrevision_5']])
        
        #mete el array de 2D en uno de 3D
        x_train[j,:,:] = muestratrainarray
        
        #se resetea el index de la muestra
        muestratrain=muestratrain.reset_index()
        muestratrain=muestratrain.drop('index', axis=1)# se modifica index
        y_train[j,:]=np.array(muestratrain.loc[34, ['Yrun', 'Yex']]) # se menten los valores de de "Yrun" e "Yex" en el array Ytrain
        
        
        j=j+1
        
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
        
        x_test[j,:,:] = muestratestarray
        
        
        
        
        muestratest=muestratest.reset_index()
        muestratest=muestratest.drop('index', axis=1)# se modifica index
        y_test[j,:]=np.array(muestratest.loc[34, ['Yrun', 'Yex']])
        
        
    
        j=j+1        
        
        
#%% 

#cargar librerias
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
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import keras.utils
from keras import utils as np_utils


import numpy as np
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
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
  

# ++++++++++ Proceso 20  +++++++++
#particionar datos en entrenamiento y validación

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=20) 
    
#%% 
# Voy a necesitar importar una serie de modulos para programar mi red neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization

#%% 
# ++++++++++++ MODELO DE PREDICCION RED NEURONAL CONVOLUCIONAL ++++++++++




from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import keras.utils
from keras import utils as np_utils


import numpy as np
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
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model

from tensorflow.keras.optimizers import Adam

# se modifican los metadatos del array insertando un nuevo eje ena tercera dimensión
x_trconv = expand_dims(x_tr, axis=3)
x_valconv  = expand_dims(x_val, axis=3)
x_testconv  = expand_dims(x_test, axis=3)
print(x_trconv.shape)
print(x_valconv.shape)
print(x_testconv.shape)

# Construccion de una red CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import callbacks
# Red feedforward API secuencial
convnet = Sequential()
# BASE MODEL
convnet.add(layers.Conv2D(64,(3,3),padding='same',input_shape=(35,9,1),activation='relu'))
convnet.add(Dropout(0.2))

convnet.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
#convnet.add(Dropout(0.25))
convnet.add(Dropout(0.1))
convnet.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
convnet.add(Dropout(0.1))
#TOP MODEL
convnet.add(layers.Flatten())
convnet.add(layers.Dense(64,activation='relu'))

convnet.add(layers.Dense(2,activation='softmax'))

convnet.compile(optimizer='adam',
                loss='categorical_crossentropy',
                
                metrics=['accuracy'])


EPOCHS = 9


fname = 'C:/Users/FernandezD3/OneDrive - Nordex SE/Documentos/Master thesis/Master/Master/tablas para modelo/Generar con el aceite resuelto/aceite resuelto2/datosmodelo3estandtotal- Solo Iso/CNN solo solo ISO/sinlubname2/checkpoint.{epoch:02d}-{val_accuracy:.2f}.keras'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(fname, monitor="val_accuracy", mode="min", save_freq="epoch", verbose=1)




H = convnet.fit(x_trconv, y_tr,shuffle=True, epochs=EPOCHS,batch_size=200, validation_data=(x_valconv , y_val), callbacks=model_checkpoint_callback)


os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\tablas para modelo\Generar con el aceite resuelto\aceite resuelto2\datosmodelo3estandtotal- Solo Iso\CNN solo solo ISO\sinlubname2")




# summarize history for accuracy
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True, linestyle='-.')
plt.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.savefig("Accuracy.png", dpi=1000.0, bbox_inches='tight', pad_inches=0.5)
plt.show()
# summarize history for loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.yscale('log')
plt.grid(True, linestyle='-.')
plt.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.savefig("Loss.png", dpi=1000.0, bbox_inches='tight', pad_inches=0.5)
plt.show()



from sklearn.metrics import classification_report
# Evaluando el modelo de predicción con las imágenes de test
print("[INFO]: Evaluando red neuronal...")
predictions = convnet.predict(x_testconv, batch_size=200)


print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1))) # las metricas se calculan con los resultados de test




y_testconvconf=y_test[:,1]

predictionsconvconf=np.where(predictions >= 0.5, 1, 0)
predictionsconvconf=predictionsconvconf[:,1]

names=["Running","Exhanged"]


#condusion matrix normalizada
cm=confusion_matrix(y_testconvconf,predictionsconvconf)
cmn = cm.astype('float')
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=names, yticklabels=names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion matrix.png",dpi=1000.0,bbox_inches='tight', pad_inches=0.5)
plt.show(block=False)
#reporte metricas de clasificacion
reportPCs = classification_report(y_testconvconf,predictionsconvconf)

sklearn.metrics.accuracy_score(y_testconvconf,predictionsconvconf)
sklearn.metrics.precision_score(y_testconvconf,predictionsconvconf)
sklearn.metrics.recall_score(y_testconvconf,predictionsconvconf)













#+++++++++++++++++++++   FIN ++++++++++++++++++++++++++++++++















#estos codigos son solo para pruebas

#%% 
#crear la tabla de chequeo
listawtgrevisados=[]
muestrapredictions=pd.DataFrame(predictions)
predictionschecking=pd.DataFrame()
predictionsconvconf=pd.DataFrame(predictionsconvconf)
y_testchecking=pd.DataFrame(y_test)

j=0
for i in test['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.         
        muestratest=test.loc[test['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        muestratest=muestratest.reset_index()
        muestratest=muestratest.drop('index', axis=1)# se modifica index
        #muestratestarray=np.array(muestratest.loc[:, ['PC_4esRob', 'PC_6esRob', 'PC_14esRob', 'FEesRob', 'ALesRob', 'CUesRob', 'PQesRob']])
        muestratest['predictionRunning']=''
        muestratest['predictionExchange']=''
        muestratest['Result']=''
        
        muestratest.loc[0,'predictionRunning']=muestrapredictions.iloc[j,0]
        muestratest.loc[0,'predictionExchange']=muestrapredictions.iloc[j,1]
        
        
        
        if predictionsconvconf.iloc[j,0]==y_testchecking.iloc[j,1]:
            muestratest.loc[0,'Result']='Ok'
        else:
            muestratest.loc[0,'Result']='Fail'
            
            
        predictionschecking = pd.concat([predictionschecking, muestratest])
        
        
            
        
        j=j+1


predictionschecking=predictionschecking.reset_index()
predictionschecking=predictionschecking.drop('index', axis=1) 




#%% 
#codigo para comprobar que funciona bien la matriz de confusion , pues caldula los fallos y aciertos en ·excahnge"
countex=0
countnoex=0
j=0 
for i in predictionschecking["Yex"]:
    if j==5074:
        print('hola')
    if i==1 and predictionschecking.loc[j,"Result"]=="Ok":
        countex=countex+1
    if i==1 and predictionschecking.loc[j,"Result"]=="Fail":
        countnoex=countnoex+1
    j=j+1

#%% 
