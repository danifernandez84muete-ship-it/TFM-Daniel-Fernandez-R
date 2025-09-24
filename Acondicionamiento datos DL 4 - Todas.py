# -*- coding: utf-8 -*-

"""
Created on Fri Aug  9 21:07:43 2024

@author: FernandezD3
"""

#++++++++++++++codigo para poner en tablamodelo, los valores de "GBXs status" en one hot encoding
reset -f
# import tensorflow as tf
# tf.__version__

import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

from pandas import Series



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

# !pip uninstall scikit-learn --yes
# !pip uninstall imblearn --yes
# !pip install scikit-learn==1.2.2
# !pip install imblearn




# ++++++++++ Proceso 11  +++++++++

# Establecemos el directorio de trabajo
import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\Modelos predictivos\Cod. acondicionamiento y tablas utilizadas")


# Comienza el proceso de Merger entre las tabla"BBDDaceites" que previamente ha sido manipulada para retirar columnas inútiles
# con la tabla "Masterdata" .Se convierten a dataframe.
#BBDDaceites = pd.read_excel(io = "BBDDaceites 02-03.xlsx", sheet_name="Sheet1")
BBDDaceitesmodelo = pd.read_excel(io = "tablamodelo3aceite.xlsx", sheet_name="Sheet1")

len(BBDDaceitesmodelo['reportWTGbis'].unique())

BBDDaceitesmodelo1=BBDDaceitesmodelo.copy()
BBDDaceitesmodelo1["Yrun"]=""
BBDDaceitesmodelo1["Yex"]=""

i=0
j=0
for i in range(len(BBDDaceitesmodelo1)):
    
    if BBDDaceitesmodelo1.loc[j,'GBXs status']== "Exchanged":
        BBDDaceitesmodelo1.loc[j,'Yrun']=0
        BBDDaceitesmodelo1.loc[j,'Yex']=1
    if BBDDaceitesmodelo1.loc[j,'GBXs status']== "Running":
        BBDDaceitesmodelo1.loc[j,'Yrun']=1
        BBDDaceitesmodelo1.loc[j,'Yex']=0
    j=j+1

BBDDaceitesmodelo1.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\pruebaspython\tablamodeloonehotencoding3.xlsx', index = False)
#++++++++++++++++++++++................................................................................................................


#%%






#++++++++++++++codigo para balancear, estandarizar los valores, contar el numero de filas de cada multiplicadora y añadir filas con 0 en las matrices 

# ++++++++++ Proceso 12  +++++++++




reset -f
import pandas as pd

# Establecemos el directorio de trabajo
import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\Modelos predictivos\Cod. acondicionamiento y tablas utilizadas")


datosmodelo = pd.read_excel(io = "tablamodeloonehotencoding3aceite.xlsx", sheet_name="Sheet1")

#cuantas multi running y exhdanged hay
run = datosmodelo.loc[datosmodelo['GBXs status'] == "Running"]
len(run['reportWTGbis'].unique())
exch = datosmodelo.loc[datosmodelo['GBXs status'] == "Exchanged"]
len(exch['reportWTGbis'].unique())


datosmodelo1 = datosmodelo.copy()
datosmodelo1estand = datosmodelo.copy()

#%%
#codigo para balancear las muestras entre running y exchange mediante eliminacin de muestras de la clase "Running"
from imblearn.under_sampling import RandomUnderSampler
datosmodelo1res=pd.DataFrame()
datosbalanceo = pd.DataFrame()
datosbalanceo=datosmodelo1[['reportWTGbis', 'Yrun', 'Yex']]
listawtgrevisados=[]
datosbalanceo=datosbalanceo.drop_duplicates()

datosbalanceo["Yexbal"]=""#tengo que poner esta fila porque si no, da error la funcion balanceo. 
#el valor de Yexbal sera 1 en las muestras de "exchanged"
      
x=datosbalanceo[['reportWTGbis','Yexbal']]


y=datosbalanceo[['Yrun','Yex']]
y=y.to_numpy()

undersample = RandomUnderSampler(sampling_strategy=0.7)#porcentaje AL REVES de datos de la clase mas imperante que queires que hya en el nuevo dataframe


xres, yres =undersample.fit_resample(x,y) 

xres["Yexbal"]=pd.DataFrame(yres)

len(xres[xres["Yexbal"]==0])
len(xres[xres["Yexbal"]==1])


#ahora en la tabla datosmodelo1 nos quedamos solo con los aeros que quedan tras hacer el resampleo


listares=xres['reportWTGbis'].unique().tolist()
for i in datosmodelo1['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    if wtg in listares:

        if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
            
            
            
            listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
            
        
            muestradatosmodelo1res=datosmodelo1.loc[datosmodelo1['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
            
            
            datosmodelo1res = pd.concat([muestradatosmodelo1res, datosmodelo1res]) 
            

len(datosmodelo1res['reportWTGbis'].unique())

#%%
# ++++++++++ Proceso 13  +++++++++
#se realiza el codificado de la variable "lubName" y "Designrevision" con BinaryEncoder


from category_encoders import BinaryEncoder

# Creamos el codificador indicandole las columnas
encoder = BinaryEncoder(cols=['lubName', 'Designrevision'])

# Ajustamos el codificador y se transforma
encoder.fit(datosmodelo1res[['lubName', 'Designrevision']])
df_binario = encoder.transform(datosmodelo1res[['lubName', 'Designrevision']])
datosmodelo1res = pd.concat([datosmodelo1res, df_binario], axis=1)

#cuantas multi running y exhdanged hay
run = datosmodelo1res.loc[datosmodelo1res['GBXs status'] == "Running"]
len(run['reportWTGbis'].unique())
exch = datosmodelo1res.loc[datosmodelo1res['GBXs status'] == "Exchanged"]
len(exch['reportWTGbis'].unique())
len(datosmodelo1res['Designrevision'].unique())


datosmodelo1res.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\tablas para modelo\Generar con el aceite resuelto\tablamodeBinaryencoding3aceite.xlsx', index = False)



#%%

# ++++++++++ Proceso 14  +++++++++

#Este proceso se encarga de revisar cuantas muestras de aceite tiene cada multiplicadora 
#y asi ayudar a establecer cual es le numero maximo de muestareas que vamos a dejar, y se establece que si tiene 35 o mas
#no se van a utilizar. Solo se utilizan aquellas que tienes menos de 35 , ya quu ehay pocas que tengs mas de 35 muestas y asi no 
#nos abliga a tener que crear matrices con tantts filas vacias.
reset -f


import pandas as pd

# Establecemos el directorio de trabajo
import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\tablas para modelo\Generar con el aceite resuelto\aceite resuelto2")


# Comienza el proceso de Merger entre las tabla"BBDDaceites" que previamente ha sido manipulada para retirar columnas inútiles
# con la tabla "Masterdata" .Se convierten a dataframe.
#BBDDaceites = pd.read_excel(io = "BBDDaceites 02-03.xlsx", sheet_name="Sheet1")
datosmodelo = pd.read_excel(io = "tablamodeBinaryencoding3aceite.xlsx", sheet_name="Sheet1")

datosmodelo1resencoding = datosmodelo.copy()




###calcular tamaño de las matrices.
numfilasdic=dict()
valoresdic=dict()
import operator
from operator import itemgetter
BBDDaceitestotal1= datosmodelo1resencoding.copy()
listawtgrevisados=[]
i=0
for i in BBDDaceitestotal1['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)

    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        
    
        muestraBBDDaceitestotal1=BBDDaceitestotal1.loc[BBDDaceitestotal1['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        
        numfilasdic[wtg] = len(muestraBBDDaceitestotal1)
        
        
val = max(numfilasdic.items(), key=itemgetter(1))

valoresdic=dict()
for valor in numfilasdic.values():
    if valor not in valoresdic:
        valoresdic[valor]=1
    else:
        valoresdic[valor]=valoresdic[valor]+1
    
    print(valor)
dfvaloresdic = pd.DataFrame(data=valoresdic, index=[0])

dfvaloresdic = (dfvaloresdic.T)       


#en este paso conviene cargarte las maquinas que tengas muchas filas y vayan a distorsionan ,en este caso borro las que tengasn mas de 36 muestras
datosmodelo1resmatriz=pd.DataFrame()
listawtgrevisados=[]
for i in BBDDaceitestotal1['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)

    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        
    
        muestraBBDDaceitestotal1=BBDDaceitestotal1.loc[BBDDaceitestotal1['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        if len(muestraBBDDaceitestotal1)<=35:
            
            datosmodelo1resmatriz = pd.concat([datosmodelo1resmatriz, muestraBBDDaceitestotal1])
            
len(datosmodelo1resencoding['reportWTGbis'].unique())
len(datosmodelo1resmatriz['reportWTGbis'].unique())




#%%
# ++++++++++ Proceso 15  +++++++++

# añadir ceros a las matrices para hacerlas de 35 filas que es la mas grande (35 variables)
i=0
j=0
#S=pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],index=['PC_4', 'PC_6','PC_14','FE', 'AL','CU','PQ','GBXs status','reportWTGbis','takenDate','Yrun','Yex','Addinol Eco Gear 320 S',	'Castrol Optigear Synthetic A 320','Castrol Optigear Synthetic X 320','Castrol Tribol 1100/320','Fuchs Gearmaster Eco 320','Fuchs Renolin Unisyn CLP 320','Klüber Klübersynth GEM 4-320 N','MOBIL SHC GEAR 320 WT','Mobilgear SHC XMP 320','Mobilgear XMP 320','Omega 690 Aceite reductora','Opet Fuchs','Total Carter SH 320	','Total SG 150','CPNHZ-180','CPNHZ-190','CPNHZ-197','CPNHZ-244','EBN 1530','EBN 1655','EBN 1785','EBN 2145','EBN 2400','EBN 2500','EBN 2980','EBN 3080','EBN 3180','EBN 3220','EBN 5460','EBN 834','EBN 916','EH904A','EH905A','GPV 510 D','GPV 535 D','PEAB 4390','PEAC 4280','PEAL 4375','PEAS 4290','PEAS 4375',	'PLH-1100','PZ3VH-126'	,'PZAB 3450', 'PZAB 3600'])
#****CUIDADO CON ESTO PORQUE!! el RandomUnderSampler puede que haya eliminado algun tipo de multiplicadora por completo , asi que el 'Designrevision_5' no estara, porque el Bynari encoder ha encodeado menos tipso de multis. Pero necesitamos que este el 'Designrevision_5', por que los modelos han sido entrenados con esta varaible tambien , y puede cambiar significativamente el resultado.
S=pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],index=['PC_4', 'PC_6','PC_14','FE','CR','SN','AL','NI','CU','PB','MO','PQ','SI','K','GBXs status','reportWTGbis','takenDate','Yrun','Yex','lubName_0','lubName_1','lubName_2','lubName_3','lubName_4','Designrevision_0','Designrevision_1','Designrevision_2','Designrevision_3','Designrevision_4','Designrevision_5'])



Sdf = pd.DataFrame(S).T
listawtgrevisados=[]
datosmodelo1ceros = pd.DataFrame()
datosmodelo1ceros2 = pd.DataFrame()
for i in datosmodelo1resmatriz['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        
    
        muestra=datosmodelo1resmatriz.loc[datosmodelo1resmatriz['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        
        muestra = muestra.sort_values('takenDate')# se ordena tabla multiplicadoras de Hugo por "Start-Up"
        muestra=muestra.reset_index()
        muestra=muestra.drop('index', axis=1)# se modifica index
        muestra2=muestra.copy()
        numfilasmuestra=0
        numfilasmuestra=len(muestra)
        Sdf['reportWTGbis']=wtg
        for j in range(numfilasmuestra, 35):# en este caso 44 es el numero de muestras mas grande que tiene una de las multiplicadoras
            #muestranew=muestra.append({'PC_4' : 0 , 'PC_6' : 0,'PC_14': 0,'GBXs status': 0,'reportWTGbis': 0,'takeDate': 0,'Yrun': 0,'Yex': 0,'PC_4es': 0,'PC_6es': 0,'PC_14es': 0},ignore_index=True)
            muestra2 = pd.concat([Sdf,muestra2], axis=0) 
            muestra = pd.concat([muestra,Sdf], axis=0) 
             
        muestra=muestra .reset_index()
        muestra=muestra.drop('index', axis=1) 
        
        muestra2=muestra2 .reset_index()
        muestra2=muestra2.drop('index', axis=1)
        #muestramatriz=muestra.to_mumpy(na_value=0)
        datosmodelo1ceros = pd.concat([datosmodelo1ceros, muestra])
        datosmodelo1ceros2 = pd.concat([datosmodelo1ceros2, muestra2])
        #reseteo el index
        datosmodelo1ceros=datosmodelo1ceros.reset_index()
        datosmodelo1ceros=datosmodelo1ceros.drop('index', axis=1)# se modifica index
        datosmodelo1ceros2=datosmodelo1ceros2.reset_index()
        datosmodelo1ceros2=datosmodelo1ceros2.drop('index', axis=1)# se modifica index
        
        
        
datosmodelo1ceros2.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\tablas para modelo\Generar con el aceite resuelto\aceite resuelto2\datosmodelo3estandtotalaceite.xlsx', index = False)
        
#%%

# ++++++++++ Proceso 16  +++++++++
#Se encargar de separar de forma aleatoria las muestras entre los conjuntos de entrenamiento y testeo. La separacion se hace en base al numero de multiplicadora , columna 'reportWTGbis'.
#Se establece un 15% de las multiplicadoras en el grpo de testeo

reset -f
import pandas as pd


import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\Modelos predictivos\Cod. acondicionamiento y tablas utilizadas")
BBDDaceitesmodelo = pd.read_excel(io = "datosmodelo3estandtotalaceite.xlsx", sheet_name="Sheet1")
datosmodelo1ceros=BBDDaceitesmodelo.copy()
#dividir los datos en train y test , con division aleatoria



#dividir los datos en train y test , con division aleatoria pero manteniendo columna 'reportWTGbis'

from sklearn.model_selection import GroupShuffleSplit 

splitter = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state = 7)
split = splitter.split(datosmodelo1ceros, groups=datosmodelo1ceros['reportWTGbis'])
train_inds, test_inds = next(split)

train = datosmodelo1ceros.iloc[train_inds]
test = datosmodelo1ceros.iloc[test_inds]

test=test.reset_index()
test=test.drop('index', axis=1)# se modifica index

train=train.reset_index()
train=train.drop('index', axis=1)# se modifica index

len(train['reportWTGbis'].unique())
len(test['reportWTGbis'].unique())

#%%
# ++++++++++ Proceso 17  +++++++++
ex=test[test["GBXs status"]=="Exchanged"]
run=test[test["GBXs status"]=="Running"]
len(ex['reportWTGbis'].unique())
len(run['reportWTGbis'].unique())

ex=train[train["GBXs status"]=="Exchanged"]
run=train[train["GBXs status"]=="Running"]
len(ex['reportWTGbis'].unique())
len(run['reportWTGbis'].unique())


#quitar maquinas exchange de test y meterlas en train
listawtgrevisados=[]
countex=0
countrun=0
# esto es un contador para saber cuantes multiplicadoras estan en "Exchange" y cuantas en "Running" en test
for i in test['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.  
        muestra=test.loc[test['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        muestra=muestra.reset_index()
        muestra=muestra.drop('index', axis=1)# se modifica index
        
        if muestra.loc[34,'Yex']==1:
            countex=countex+1
        
        else:
            countrun=countrun+1



listawtgrevisados=[]
countex=0
countrun=0

#esto resaliza el elinimado de las muestras correspondientes 50 multiplicadoras en estado "Exchange" en el grupo test
#y se añaden al grupo de train
for i in test['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.  
        muestra=test.loc[test['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        muestra=muestra.reset_index()
        muestra=muestra.drop('index', axis=1)# se modifica index
        
        if muestra.loc[34,'Yex']==1:
            countex=countex+1
            if countex<50:
                train = pd.concat([train, muestra])#se van concatenando las muetras resultantes creadas
                test=test[test['reportWTGbis']!=wtg]#se eliminan de test las muestas correspondientes a esa multiplicadora.
            
        
        else:
            countrun=countrun+1
            
        

train=train.reset_index()
train=train.drop('index', axis=1)# se modifica index       

test=test.reset_index()
test=test.drop('index', axis=1)# se modifica index   

#esto es un contador nuevamente como el anterior para saber cuantes multiplicadoras estan en "Exchange" y cuantas en "Running" en test

listawtgrevisados=[]
countex=0
countrun=0    

for i in test['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.  
        muestra=test.loc[test['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
        muestra=muestra.reset_index()
        muestra=muestra.drop('index', axis=1)# se modifica index
        
        if muestra.loc[34,'Yex']==1:
            countex=countex+1
        
        else:
            countrun=countrun+1


ex=train[train["GBXs status"]=="Exchanged"]
run=train[train["GBXs status"]=="Running"]
len(ex['reportWTGbis'].unique())
len(run['reportWTGbis'].unique())



#%%

# ++++++++++ Proceso 18  +++++++++
#standarizacion de valores numericos 
#esta estandarizacio StandardScaler no es buena por que los valores que tenenos no siguen distribucion normal y tenen muchos sesgo( tienes muy altos considrasdos outliers9)



from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from operator import itemgetter

trainS = train.iloc[:, 0:14]
testS=test.iloc[:, 0:14]






#he quitado la Normalizacion

#estandarizacion con metodo RobustScaler

from sklearn.preprocessing import RobustScaler



x_scaler1 = RobustScaler()


df11train =x_scaler1.fit_transform(trainS)
df11test = x_scaler1.transform(testS)

df11train = pd.DataFrame(df11train, columns=['PC_4', 'PC_6', 'PC_14','FE','CR','SN','AL','NI','CU','PB','MO','PQ','SI','K'])
train['PC_4esRob']=''
train['PC_4esRob']=df11train['PC_4']
train['PC_6esRob']=''
train['PC_6esRob']=df11train['PC_6']
train['PC_14esRob']=''
train['PC_14esRob']=df11train['PC_14']
train['FEesRob']=''
train['FEesRob']=df11train['FE']
train['CResRob']=''
train['CResRob']=df11train['CR']
train['SNesRob']=''
train['SNesRob']=df11train['SN']
train['ALesRob']=''
train['ALesRob']=df11train['AL']
train['NIesRob']=''
train['NIesRob']=df11train['NI']
train['CUesRob']=''
train['CUesRob']=df11train['CU']
train['PBesRob']=''
train['PBesRob']=df11train['PB']
train['MOesRob']=''
train['MOesRob']=df11train['MO']
train['PQesRob']=''
train['PQesRob']=df11train['PQ']
train['SIesRob']=''
train['SIesRob']=df11train['SI']
train['KesRob']=''
train['KesRob']=df11train['K']








df11test = pd.DataFrame(df11test, columns=['PC_4', 'PC_6', 'PC_14','FE','CR','SN','AL','NI','CU','PB','MO','PQ','SI','K'])

test['PC_4esRob']=''
test['PC_4esRob']=df11test['PC_4']
test['PC_6esRob']=''
test['PC_6esRob']=df11test['PC_6']
test['PC_14esRob']=''
test['PC_14esRob']=df11test['PC_14']
test['FEesRob']=''
test['FEesRob']=df11test['FE']
test['CResRob']=''
test['CResRob']=df11test['CR']
test['SNesRob']=''
test['SNesRob']=df11test['SN']
test['ALesRob']=''
test['ALesRob']=df11test['AL']
test['NIesRob']=''
test['NIesRob']=df11test['NI']
test['CUesRob']=''
test['CUesRob']=df11test['CU']
test['PBesRob']=''
test['PBesRob']=df11test['PB']
test['MOesRob']=''
test['MOesRob']=df11test['MO']
test['AGesRob']=''
test['PQesRob']=df11test['PQ']
test['SIesRob']=''
test['SIesRob']=df11test['SI']
test['KesRob']=''
test['KesRob']=df11test['K']




train.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\tablas para modelo\Generar con el aceite resuelto\aceite resuelto2\traindatosmodelo3estandtotalaceite.xlsx', index = False)
        

test.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\tablas para modelo\Generar con el aceite resuelto\aceite resuelto2\testdatosmodelo3estandtotalaceite.xlsx', index = False)
        

#%%

        

#%% 
