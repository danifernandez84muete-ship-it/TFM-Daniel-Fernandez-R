# -*- coding: utf-8 -*-
"""
Created on May 24 22:58:58 2024

@author: FernandezD3
"""

#++++++BLOQUE 1 ++++++++++++++  HACE EL MERGER ENTRE LA TABLA "BBDDaceites 02-03.xlsx" QUE CONTIENE TODOS LOS ANALISIS DE LAS MUESTRAS DE ACEITE YLA TABLA "MASTER DATA" 
#QUE CONTIENE IFOMRACION DE LOS AEROGENERADORES

# ++++++Proceso 1 +++++
#SE CARGAN LAS FUNCIONES Y LA TABLA "BBDDaceites.xlsx" QUE CONTIENE TODOS LOS ANALISIS DE LAS MUESTRAS DE ACEITE.
# TABMIEN SE CARGA LA TABLA "MASTER DATA" DONDE ESTA LA INFOMRACINO RELATIVA A LOS AEROGENERADORES

reset -f

import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

# Establecemos el directorio de trabajo
import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos")


# Comienza el proceso de Merger entre las tabla"BBDDaceites" que previamente ha sido manipulada para retirar columnas inútiles
# con la tabla "Masterdata" .Se convierten a dataframe.
#BBDDaceites = pd.read_excel(io = "BBDDaceites 02-03.xlsx", sheet_name="Sheet1")
Masterdata = pd.read_excel(io = "Master Data.xlsx", sheet_name="New Report")
BBDDaceites = pd.read_excel(io = "BBDD aceites.xlsx", sheet_name="New Report")

#%%


# ++++++++++ Proceso 2  +++++++++

#hago copia de las BBDDs para no cargarlas todo el rato y creo columnas en la tabla BBDDaceites
BBDDaceites1 = BBDDaceites.copy()
Masterdata1 = Masterdata.copy()



# el proceso va a ser añadir columnas a la tabla "BBDDaceites1", se añaden ahora como vacias
BBDDaceites1["Turbine Type"]=""
BBDDaceites1["Turbine Generation"]=""
BBDDaceites1["class"]=""
BBDDaceites1["reportingFarmName"]=""

lista=[]


def isNumeric(s): #funcion que identifica si un valor es numerico o no
    try:
        int(s)
        return True
    except ValueError:
        return False

#%%

# +++++++++++  Proceso  3 ++++++++++++
# enriquecer tabla de BBDD aceites con columnas de tabla MasterData
cont=0
for i in range(len(BBDDaceites1)): #se van recorriendo todas la filas de la tabla "BBDDaceites" , por nombre de aerogenerador
    
    BBDDwtg=BBDDaceites1.loc[i, "reportWTG"]
    
    if isNumeric(BBDDwtg) == True: #chequea si el nombre del aero esta formato numérico
        result = Masterdata1.loc[Masterdata1['WTG Number'] == int(BBDDwtg)] #coge muestra de tabla "Masterdata1" , de
        #valores que coinciden con el nombre del aero.
        
    else:
     
        result = Masterdata1.loc[Masterdata1['WTG Number'] == (BBDDwtg)] #coge muestra de tabla "Masterdata1" , de
        #valores que coinciden con el nombre del aero.
        


  
    
    
    
    if len(result)!=0: #verifica que el aero que ha cogido de "BBDDaceites1", tiene valores en la tabla "Masterdata1"
          #añade las columnas que va a añadir de tabla "Masterdata1"
        BBDDaceites1.loc[i, "Turbine Type"] = result.iloc[0, 1]
        BBDDaceites1.loc[i, "Turbine Generation"] = result.iloc[0, 2]
        BBDDaceites1.loc[i, "class"] = result.iloc[0, 3]
        BBDDaceites1.loc[i, "reportingFarmName"] = result.iloc[0, 21]
        cont=cont+1
        
        
    else: # en caso de no tener valores la tabla "Masterdata1"se registra que este aero ya ha sido chequeado, 
    #asi no repetimos todo el rato este proceso en la tabla "BBDDacites"
        lista.append(BBDDwtg)
        
        BBDDaceites1.drop(i, axis=0, inplace=True)
       
    



BBDDaceites1.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos\BBDDaceites-V1.xlsx', index = False)





#%%    


#+++++++++BLOQUE 2 +++++
 
 # Merge entre "BBDDD aceites" con "GBX failure overview-V1". Se asignan multiplicadora y su historico a la tabla 
 #BBDDaceites-V1-ML.xlsx, usando la tabla Gbx FAilure overview.xlsm.
 
# +++++++ Proceso 4  +++++++

#Se vuelven a cargar las funciones necesarias y se cargan las dos tablas.
  

reset -f

import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

# Establecemos el directorio de trabajo
import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos")


    
BBDDaceites = pd.read_excel(io = "BBDDaceites-V1-ML.xlsx", sheet_name="Sheet1")
Gbxs = pd.read_excel(io = "Gbx FAilure overview-V1.xlsm", sheet_name="Sheet1")

#%%

#+++   Proceso  5 ++++

#Crea nuevas columnas un la tabla "BBDDaceites-V1-ML.xlsx" con información de multiplicadoras que luego irá rellenando.
#Va cogiendo las muestras de aceite por aerogenerador de la tabla "BBDDaceites-V1-ML.xlsx" y le va asignando  la información correspondiente de la multiplicadora.
# Esta información la coge de la tabla "Gbx FAilure overview-V1.xlsm"
#La forma de ir juntando esta información de estas dos tablas, es fijándose en la fecha de puesta en marcha de la multiplicadora y contrastarla con la fecha de toma de la muestra de aceite.

#####IMPORTANTE ### SE HAN REMOVIDO ESTAS MEDAS DE LA TABLA Gb falure overview " por que si no da erro
# son las que tienen la columna Start_up Date= datos de 1900
  

import math
i=0
listawtgrevisados=[]
BBDDaceites1 = BBDDaceites.copy()
Gbxs1 = Gbxs.copy()
BBDDaceites1 = BBDDaceites1.sort_values('reportWTG')# ordeno BBDD acietes por "reportWTG"
BBDDaceites1=BBDDaceites1.reset_index()#resetel la columan index
BBDDaceites1=BBDDaceites1.drop('index', axis=1)#se borra la angigua columna index
valores_columnas = list(BBDDaceites1.columns.values)
muestraBBDDaceitestotal = pd.DataFrame()
for i in range(len(valores_columnas)):
    muestraBBDDaceitestotal[valores_columnas[i]]=""
    
    
    
    
rep=0
wtg1=0
for i in BBDDaceites1['reportWTG']:#va iterando en la tabla ordenada de "BBDDaceites" por nombre de aerogenerador.
    
    wtg=i
    muestraBBDDaceites1=BBDDaceites1.loc[BBDDaceites1['reportWTG'] == wtg] #se coge muestra de la tabla "BBDDaceites" 
                                                                                #según el nombre del aerogenerador.
    
    
    result = Gbxs.loc[Gbxs['Superord.WTG'] == (wtg)]#coje una muestra de la tabla"Gbx FAilure overview" que se corresponden
    #con el aero que se esta revisando de "BBDDaceites" 
    muestraGbxs = result.sort_values('Start-up date')# ordeno BBDD acietes por "reportWTG"
    muestraGbxs=muestraGbxs.reset_index()#puendo nuevo indexretiro el index
    muestraGbxs=muestraGbxs.drop('index', axis=1)
    muestraGbxs["Start-up date"]=pd.to_datetime(muestraGbxs["Start-up date"])#modifica esta variable a fecha para que no de 
                                                                             #para que no de errores
        
 
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        rep=0
        listawtgrevisados.append(wtg)
           

  
  
    if (len(muestraGbxs)!=0) and (rep==0):# condicional de que la muesta de la tabal "Gbx FAilure overview" no este vacia,
                                          #y no se haya repetido el aero chequeado
        rep=1
        #if pd.isnull(muestraGbxs.loc[0,'Start-up date']): 
        #    print('hola')
        #se añaden las columns a la muestra de la tabla "BBDDaceites1" que se van a añadir de "Gbx FAilure overview"
        muestraBBDDaceites1["takenDate"]=pd.to_datetime(muestraBBDDaceites1["takenDate"])#variable "takenDate" se pone de tipo fecha
        muestraBBDDaceites1=muestraBBDDaceites1.sort_values(by='takenDate',ascending= True)#columna "takenDate" se ordena desdendente
        muestraBBDDaceites1=muestraBBDDaceites1.reset_index()
        muestraBBDDaceites1=muestraBBDDaceites1.drop('index', axis=1)
        muestraBBDDaceites1["Material"]=""
        muestraBBDDaceites1["Design revision"]=""
        muestraBBDDaceites1["Failed"]=""
        
        muestraBBDDaceites1["Failure date"]=""
        muestraBBDDaceites1["Failed Subcomponent"]=""
        muestraBBDDaceites1["Technical issue"]=""
        muestraBBDDaceites1["Failure Mode"]=""
        
        muestraBBDDaceites1["Exchanged"]=""
        muestraBBDDaceites1["Exchange date"]=""
        muestraBBDDaceites1["Up-tower component exchanged"]=""
        muestraBBDDaceites1["Uptower repair date"]=""
        muestraBBDDaceites1["Start-up date"]=""
        muestraBBDDaceites1["GBXs status"]=""
        muestraBBDDaceites1["reportWTGbis"]=""

        
        m=0
        n=0
        for j in muestraGbxs['Start-up date']: #recorre la muestra de "Gbx FAilure overview", por fecha de muestra en marcha 
                                                #de antigua a nueva
            type(j)
            m=0
            
            if not 'datetime.datetime' in str(type(j)):# si no hay fecha o es es ilegible , se pasa esa esa multiplicadora
                                                         #por no tener fecha. y pasa a la siguiete mutiplicadora de la muestra.
     
                print('yes')
                continue
            
            
            for k in muestraBBDDaceites1['takenDate']: #recorre la muestra la tabla "BBDDaceites" por fecha de toma
                
                if k>=j:# si la fecha de de puesta en marcha de la fila de la muestra de "Gbx FAilure overview" es anterior
                        #que la fecha de toda de muestras de aceites, permite rellenar las columnas de fila que se han añadido.
                    
                    
                    muestraBBDDaceites1.loc[m,"Material"]=muestraGbxs.loc[n,"Material"]
                    muestraBBDDaceites1.loc[m,"Design revision"]=muestraGbxs.loc[n,"Design revision"]
                    muestraBBDDaceites1.loc[m,"Failed"]=muestraGbxs.loc[n,"Failed"]
                    muestraBBDDaceites1.loc[m,"Failure date"]=muestraGbxs.loc[n,"Failure date"]
                    muestraBBDDaceites1.loc[m,"Failed Subcomponent"]=muestraGbxs.loc[n,"Failed Subcomponent"]
                    muestraBBDDaceites1.loc[m,"Technical issue"]=muestraGbxs.loc[n,"Technical issue"]
                    muestraBBDDaceites1.loc[m,"Failure Mode"]=muestraGbxs.loc[n,"Failure Mode"]
                    muestraBBDDaceites1.loc[m,"Exchanged"]=muestraGbxs.loc[n,"Exchanged"]
                    
                    muestraBBDDaceites1.loc[m,"Exchange date"]=muestraGbxs.loc[n,"Exchange date"]
                    muestraBBDDaceites1.loc[m,"Up-tower component exchanged"]=muestraGbxs.loc[n,"Up-tower component exchanged"]
                    muestraBBDDaceites1.loc[m,"Uptower repair date"]=muestraGbxs.loc[n,"Uptower repair date"]
                    muestraBBDDaceites1.loc[m,"Start-up date"]=muestraGbxs.loc[n,"Start-up date"]
                    muestraBBDDaceites1.loc[m,"GBXs status"]=muestraGbxs.loc[n,"GBXs status"]
                    # se añade una columna con el nombre del aero y una letra , para diferenciar cada vez que 
                    # que hay una multiplicadora distinta en un aero , ya que el nombre de la multiplicadora puede
                    #ser el mismo.
                    if n==0: 
                        muestraBBDDaceites1.loc[m,"reportWTGbis"]=str(muestraBBDDaceites1.loc[m,"reportWTG"])+"A"
                    if n==1: 
                        muestraBBDDaceites1.loc[m,"reportWTGbis"]=str(muestraBBDDaceites1.loc[m,"reportWTG"])+"B"
                    if n==2: 
                        muestraBBDDaceites1.loc[m,"reportWTGbis"]=str(muestraBBDDaceites1.loc[m,"reportWTG"])+"C"
                    if n==3: 
                        muestraBBDDaceites1.loc[m,"reportWTGbis"]=str(muestraBBDDaceites1.loc[m,"reportWTG"])+"D"
                    if n==4: 
                        muestraBBDDaceites1.loc[m,"reportWTGbis"]=str(muestraBBDDaceites1.loc[m,"reportWTG"])
                    if n==5: 
                        muestraBBDDaceites1.loc[m,"reportWTGbis"]=str(muestraBBDDaceites1.loc[m,"reportWTG"])
                    
                
                m=m+1 # pasa la siguiente fila de la muestra de BBDD acietes
                    
            n=n+1 #pasa a la siguiente fila de la muestra de "Gbx FAilure overview"
                          
      
        muestraBBDDaceitestotal = pd.concat([muestraBBDDaceitestotal, muestraBBDDaceites1]) 
    
    else:
        
        
        continue
        
            
#muestraBBDDaceitestotal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\pruebaspython\BBDDaceitestotal.xlsx', index = False)
muestraBBDDaceitestotal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos\BBDDaceites-V2.xlsx', index = False)



#%%    



 #+++++++++  BLOQUE 3 +++++
 
 # Ahora se revisa del dataframe "BBDDaceites-V2" que se genero en el BLOQUE 2 para identificar las filas 
 #  de las muestras de aciete, que tienen columnas de informacion de mutliplicadora no completdas (sin asigna multiplicadora y su historico),
 #  y ver se pueden completar con la tabla de Hugo  "MGB_EXPORT_ZCS_EQREP_02_Feb2024", que tambien contiene infomracion de algunas multiplcadoras.
 #
 
 
 #+++Proceso 6 ++++

#Carga las tablas "BBDDaceites-V2" y "MGB_EXPORT_ZCS_EQREP_02_Feb2024" , y ordenar por aerogenerador la de acietes.

reset -f

import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys

# Establecemos el directorio de trabajo
import os
#os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis")
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos")



#se cogen tablas de BBDDaceitestotal.xlsx y la de MGB_EXPORT_ZCS_EQREP_02_Feb2024.xlsx

BBDDaceitestotal = pd.read_excel(io = "BBDDaceites-V2.xlsx", sheet_name="Sheet1")
BBDDaceitestotal2 = BBDDaceitestotal.copy()
MGBx = pd.read_excel(io = "MGB_EXPORT_ZCS_EQREP_02_Feb2024.xlsx", sheet_name="MGB_EXPORT_ZCS_EQREP_02_Feb2024")
MGBx1 = MGBx.copy()

#BBDDaceitestotal2 = BBDDaceitestotal.copy()




BBDDaceitestotal2 = BBDDaceitestotal2.sort_values('reportWTG')# ordeno BBDD aceites por "reportWTG"
BBDDaceitestotal2=BBDDaceitestotal2.reset_index()
BBDDaceitestotal2=BBDDaceitestotal2.drop('index', axis=1) #se modifica el index
muestraBBDDaceitestotalfinal = pd.DataFrame()
import math
i=0
listawtgrevisados=[]


#%%   

#+++Proceso 7 ++++

# se encarga de ir recorriendo la tabla de acietes por muestas  y por areogenrador, y a la vez recorre la tabla de las 
#multiplicaoras para ver si hay multiplicadoras que coincidan con las fechas de las tomas de las muestras de acietes. 
# lo que hace es comparar la fecha de puesta en marcha de la multiplicadora con la fecha de toma de la muestra de aceite. 
#y en caso de que la fecha de puesta en marcha de una multiplicadora sea anterior a una fecha de toma de aceite , pues se le asigna la 
#infomracion relativa a esa multipliadora . 

 
i=0
listawtgrevisados=[]
muestraBBDDaceitestotalfinal = pd.DataFrame()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in BBDDaceitestotal2['reportWTG']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
                                        
    primero=0
    wtg=i
    muestraBBDDaceitestotal2=BBDDaceitestotal2.loc[BBDDaceitestotal2['reportWTG'] == wtg]#coge muestra de tabla BBDDaceites
                                                                                            #por el nombre de aero.
    
    
  
    
    result = MGBx1.loc[MGBx1['SuperWTG'] == (wtg)]#coges muetra de tabla "MGB_EXPORT_ZCS_EQREP_02_Feb2024" por nombre de aero
    muestraMGBx1 = result.sort_values('Start-Up')# se ordena tabla multiplicadoras de Hugo por "Start-Up"
    muestraMGBx1=muestraMGBx1.reset_index()
    muestraMGBx1=muestraMGBx1.drop('index', axis=1)# se modifica index
    muestraMGBx1["Start-Up"]=pd.to_datetime(muestraMGBx1["Start-Up"])# se estandarizan las fechas para evitar errores
        
 
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        
        rep=0
        listawtgrevisados.append(wtg)
           


        
  
  
    if (len(muestraMGBx1)!=0) and (rep==0):# condicional de que la muesta de la tabal "MGB_EXPORT_ZCS_EQREP_02_Feb2024" no este vacia,
                                          #y no se haya repetido el aero chequeado
        rep=1
       
        
        muestraBBDDaceitestotal2["takenDate"]=pd.to_datetime(muestraBBDDaceitestotal2["takenDate"])#variable "takenDate" se pone de tipo fecha
        muestraBBDDaceitestotal2=muestraBBDDaceitestotal2.sort_values(by='takenDate',ascending= True)#columna "takenDate" se ordena desdendente
        muestraBBDDaceitestotal2=muestraBBDDaceitestotal2.reset_index()
        muestraBBDDaceitestotal2=muestraBBDDaceitestotal2.drop('index', axis=1)
        m=0
        n=0
        for j in muestraMGBx1['Start-Up']:#recorre la muestra de "MGB_EXPORT_ZCS_EQREP_02_Feb2024", por fecha de muestra en marcha 
                                                #de antigua a nueva
            type(j)
            m=0
            
            if not "<class 'pandas._libs.tslibs.timestamps.Timestamp'>" in str(type(j)):# si no hay fecha o es es ilegible , se pasa esa esa multiplicadora
                                                         #por no tener fecha. y pasa a la siguiete mutiplicadora de la muestra.
     
                print('yes')
                continue
            
            
            for k in muestraBBDDaceitestotal2['takenDate']:#recorre la muestra la tabla "BBDDaceites" por fecha de toma
                
                if k>=j and muestraBBDDaceitestotal2.isnull().loc[m,"reportWTGbis"]:# si la fecha de de puesta en marcha de la fila de la muestra de "MGB_EXPORT_ZCS_EQREP_02_Feb2024" es anterior
                        #que la fecha de toda de muestras de aceites, permite rellenar las columnas de fila que se han añadido.
                    
                    #se can llenando las columnas de la tabla BBDDaceites con las de "MGB_EXPORT_ZCS_EQREP_02_Feb2024"
                    muestraBBDDaceitestotal2.loc[m,"Material"]=muestraMGBx1.loc[n,"Material"]
                    muestraBBDDaceitestotal2.loc[m,"Design revision"]=muestraMGBx1.loc[n,"Description of Technical Object"]
                    if primero==0:# si se detecta que es la primera multi que se mete y no habia otra antes , se cataloga como 
                                 #dañada por que , se sabes que la multiplicadora fue cambiada por un fallo.
                        muestraBBDDaceitestotal2.loc[m,"Failed"]="Failed"
                    
                    muestraBBDDaceitestotal2.loc[m,"Start-up date"]=muestraMGBx1.loc[n,"Start-Up"]
                    #se asignan a la nueva columna de nobres , el nombre del aero mas una letra H para saber que este multiplicadora
                    # viene de la tabla de hugo
                    if n==0: 
                        muestraBBDDaceitestotal2.loc[m,"reportWTGbis"]=str(muestraBBDDaceitestotal2.loc[m,"reportWTG"])+"H"
                    if n==1: 
                        muestraBBDDaceitestotal2.loc[m,"reportWTGbis"]=str(muestraBBDDaceitestotal2.loc[m,"reportWTG"])+"H2"
                    if n==2: 
                        muestraBBDDaceitestotal2.loc[m,"reportWTGbis"]=str(muestraBBDDaceitestotal2.loc[m,"reportWTG"])+"H3"
                    
                
                
                
                else:
                    primero=1
                    
                m=m+1  # variable para mover a la siguiente fila de   muestraBBDDaceitestotal2
            n=n+1# variable para mover a la siguiente fila de muestraMGBx1
                          
       
        
        

        muestraBBDDaceitestotalfinal = pd.concat([muestraBBDDaceitestotalfinal, muestraBBDDaceitestotal2])
    
    else:
        
        
        continue
    
 
#muestraBBDDaceitestotalfinal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\muestraBBDDaceitestotalfinal.xlsx', index = False)

muestraBBDDaceitestotalfinal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos\BBDDaceites-V3.xlsx', index = False)


#%% 


#+++Proceso 8 ++++   

# este script es utilizado para solucionar una problema en la columna "lastChangeDate". Y hacer que la fecha del ultimo cambio 
#de aceite estuviese en la misma fila que la toma mas reciente despues del cambio de aceite. Asi tener trazado cuando se hacen los cambios
# de acite por si se utilizan . La tabla de salida es "BBDDaceites-V4.xlsx"


reset -f

from datetime import datetime
import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys
muestraBBDDaceitestotalfinal = pd.DataFrame()
listawtgrevisados=[]
listadate=[]
# Establecemos el directorio de trabajo
import os
#os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master")
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos")

BBDDaceitestotal = pd.read_excel(io = "BBDDaceites-V3.xlsx", sheet_name="Sheet1")    

#BBDDaceitestotal = pd.read_excel(io = "tarea lastchangedata.xlsx", sheet_name="Sheet1")
BBDDaceitestotal1 = BBDDaceitestotal.copy()




for i in BBDDaceitestotal1['reportWTG']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    if wtg==84699:
        print ('hola')
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        p=0 #variable para una vez que ha asigando una fecha last changedate a una fila , no lo vuelva a hacer mas. entro del bucle de ir asignadondo 
        listadate=[]#lista para que una vez que a asignado una fecha last changedate a una fila , no lo vuelva a hacer mas. 
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        primero=0
    
        muestraBBDDaceitestotal1=BBDDaceitestotal1.loc[BBDDaceitestotal1['reportWTG'] == wtg]#coge muestra de tabla BBDDaceites
                                                                                            #por el nombre de aero.
    
        muestraBBDDaceitestotal1 =  muestraBBDDaceitestotal1.sort_values('takenDate')# se ordena tabla multiplicadoras de Hugo por "Start-Up"
        muestraBBDDaceitestotal1=muestraBBDDaceitestotal1.reset_index()
        muestraBBDDaceitestotal1=muestraBBDDaceitestotal1.drop('index', axis=1)# se modifica index
        muestraBBDDaceitestotal1["takenDate"]=pd.to_datetime(muestraBBDDaceitestotal1["takenDate"])# se estandarizan las fechas para evitar errores
        muestraBBDDaceitestotal1["lastChangeDate"]=pd.to_datetime(muestraBBDDaceitestotal1["lastChangeDate"])#se estandarizan las fechas para evitar errores
        muestraBBDDaceitestotal2 = muestraBBDDaceitestotal1.copy()
        muestraBBDDaceitestotal22=muestraBBDDaceitestotal1.copy()#esta es la tabla muestra que ira rellenando
        muestraBBDDaceitestotal22=muestraBBDDaceitestotal22.drop ('lastChangeDate', axis = 1)#borro esta columna
       
        
        import matplotlib.pyplot as plt
        
        E=muestraBBDDaceitestotal1.plot.scatter(x='PC_4',y='takenDate')
        
        #for idx, row in BBDDaceites1.iterrows():
           
        #    E.annotate(row['GBXs status'], (row['PC_4'], row['PC_6']))
        
        plt.show()
        
        muestraBBDDaceitestotal22["lastChangeDate"]=""#creo otra columna vacia
        for j in range(len(muestraBBDDaceitestotal1)):#recorre la muestra de muestraBBDDaceitestotal1
            m=0
            p=0
            changedate=muestraBBDDaceitestotal1.loc[j,"lastChangeDate"]#coje la feha "lastChangeDate".
            
            changedatecheck=str(changedate.to_pydatetime())#convierte la fecha en string.
            
            if changedatecheck=='1900-01-01 00:00:00':# si es esta de 1900 fecha la quita y pone vacio
                muestraBBDDaceitestotal22.loc[j,"lastChangeDate"]=''
                continue
            if changedate in listadate: #si esta fehca ya ha sido revisada y colocada se pasa de ella
                continue
            for m in range(len(muestraBBDDaceitestotal1)):#recorre la "muestraBBDDaceitestotal1" la columna "takenDate" , para ver en que fila coloca la fecha "lastChangeDate".
                takendate=muestraBBDDaceitestotal1.loc[m,"takenDate"]#coge la fecha "takenDate" de la fila
                if changedate<=takendate and p==0:# si la feha de toma es mayor que la de ultimo cambio de aciete entra
                    muestraBBDDaceitestotal22.loc[m,"lastChangeDate"]=changedate# se asigna la fecha de ultimo cambio a la fila de la columan ultimo cambio
                    p=1#variable para que no se vuelva a asinar esta fecha de ultimo cambio a ninguna fila mas.
                    listadate.append(changedate)#vaialbe para que la feha de ultimo cambio se guarde en una lista y setenga contancia de que ya fue asignada a una fila


        for l in range(len(muestraBBDDaceitestotal22)):#con este bucle se modifica las fechjas 1900 a vacio por si hubises quedado alguna
            changedate= muestraBBDDaceitestotal22.loc[l,"lastChangeDate"]
            
                
            if changedate=='1900-01-01 00:00:00.000':
                muestraBBDDaceitestotal22.loc[j,"lastChangeDate"]=''
                
            # elif  pd.isnull(changedate):
            #     continue
        
            # else:
                
            #     date_str = muestraBBDDaceitestotal2.loc[l,"lastChangeDate"]#'2023-02-28 14:30:00'
            #     date_format = '%Y-%m-%d %H:%M:%S'

            #     date_obj = datetime.strptime(date_str, date_format)
            #     print(date_obj)

                
            


    
        muestraBBDDaceitestotalfinal = pd.concat([muestraBBDDaceitestotalfinal, muestraBBDDaceitestotal22])#se van concatenando las muetras resultantes creadas

#muestraBBDDaceitestotalfinal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\solucion columna lastchangedate.xlsx', index = False)
    
muestraBBDDaceitestotalfinal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos\BBDDaceites-V4.xlsx', index = False) 


#%%


#+++Proceso 9 ++++   

# este script es utilizado para solucionar unuevamente un problema. Como es quitar muestas de aceite  con la fecha de toma repetidas "takenDate", y dejar la medida que tenga un valor de "PC_4" mayor. 
#Tambien se reitran todas las muestras que para una misma multiplicadora, tenemos menos de 4 muestras. Por se consideró 
#que menos de 4 muestras no va a ser representativo para generar un modelo predictivo. La tabla de salida es "BBDDaceites-V5.xlsx"


reset -f
from datetime import datetime
import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys
muestraBBDDaceitestotalfinal = pd.DataFrame()
listawtgrevisados=[]
listadate=[]
# Establecemos el directorio de trabajo
import os
os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos")


    

BBDDaceitestotal = pd.read_excel(io = "BBDDaceites-V4.xlsx", sheet_name="Sheet1")
BBDDaceitestotal1 = BBDDaceitestotal.copy()




for i in BBDDaceitestotal1['reportWTGbis']:#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=i
    print(wtg)
    if (wtg)=="87852B":
        print ("ya")
    if not wtg in listawtgrevisados:# se registran los nombres de aeros en una lista para que no se repitan
        p=0 #variable para una vez que ha asigando una fecha last changedate a una fila , no lo vuelva a hacer mas. entro del bucle de ir asignadondo 
        
        
        listawtgrevisados.append(wtg) #añado este aero a la lista para no repertirlo mas.                                   
        primero=0
    
        muestraBBDDaceitestotal1=BBDDaceitestotal1.loc[BBDDaceitestotal1['reportWTGbis'] == wtg]#coge muestra de tabla BBDDaceites
                                                                                            #por el nombre de aero.
    
        
        muestraBBDDaceitestotal1 =  muestraBBDDaceitestotal1.sort_values('takenDate')# se ordena tabla multiplicadoras de Hugo por "Start-Up"
        muestraBBDDaceitestotal1=muestraBBDDaceitestotal1.reset_index()
        muestraBBDDaceitestotal1=muestraBBDDaceitestotal1.drop('index', axis=1)# se mod
       
        muestraBBDDaceitestotal1.sort_values('PC_4', inplace=True)#rutina para quitar los duplicados "takenDate" y quedarse con la fila que tenga un mayor valor de "PC_4"
        muestraBBDDaceitestotal1.drop_duplicates('takenDate', keep="last", inplace=True)
        muestraBBDDaceitestotal1.sort_index(inplace=True) 
        
        muestraBBDDaceitestotal1 =  muestraBBDDaceitestotal1.sort_values('takenDate')# se ordena tabla multiplicadoras de Hugo por "Start-Up"
        muestraBBDDaceitestotal1=muestraBBDDaceitestotal1.reset_index()
        muestraBBDDaceitestotal1=muestraBBDDaceitestotal1.drop('index', axis=1)# 
        
        if len(muestraBBDDaceitestotal1)>=4: # Si hay menos de 4 muestras se desechan las muestras
       
        
            muestraBBDDaceitestotalfinal = pd.concat([muestraBBDDaceitestotalfinal, muestraBBDDaceitestotal1])#se van concatenando las muetras resultantes creadas
    
    
        else:
            continue
        
        
muestraBBDDaceitestotalfinal.to_excel(r'C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master\Material para entregar\purbas tablas de datos\BBDDaceites-V5.xlsx', index = False)
    
#%%
#+++Proceso 10 ++++   

# este script se utiliza para unir la tablas "BBDDaceites-V6-ML-Hugomultis.xlsx" y "BBDDaceites-V6-ML-Mikellubricantes.xlsx"
#de aceite estuviese en la misma fila que la toma mas reciente despues del cambio de aceite. Asi tener trazado cuando se hacen los cambios
# de acite por si se utilizan . La tabla de salida es "BBDDaceites-V6-ML-Fin.xlsx"


reset -f

from datetime import datetime
import seaborn as sns
import pandas as pd
import pandas as drop
import numpy as np
from datetime import datetime 
import sys
muestraBBDDaceitestotalfinal = pd.DataFrame()
listawtgrevisados=[]
listadate=[]
# Establecemos el directorio de trabajo
import os
#os.chdir(r"C:\Users\FernandezD3\OneDrive - Nordex SE\Documentos\Master thesis\Master\Master")
os.chdir(r"C:\Users\Usuario\Documents\Master\arreglar lo de aceites\pruebaspython")

BBDDaceitestotalHugomultis = pd.read_excel(io = "BBDDaceites-V6-ML-Hugomultis.xlsx", sheet_name="Sheet1") 
BBDDaceitestotalMikelaceites= pd.read_excel(io = "BBDDaceites-V6-ML-Mikellubricantes.xlsx", sheet_name="Sheet1")    


BBDDaceitestotal1=BBDDaceitestotalHugomultis.copy()


#%%
wtg=''
takenDate=''
lubname=''
i=0
j=0
for i in range(len(BBDDaceitestotal1)):#va iterando en la tabla de BBDDaceites por nombre de aerogenerador
    wtg=BBDDaceitestotalMikelaceites.loc[i,"reportWTGbis"]
    takenDate=BBDDaceitestotalMikelaceites.loc[i,"takenDate"]
    lubname=BBDDaceitestotalMikelaceites.loc[i,"lubName"]
    
    print(wtg)
    j=0
    for j in range(len(BBDDaceitestotal1)):
            if BBDDaceitestotalHugomultis.loc[j,"reportWTGbis"]==wtg and BBDDaceitestotalHugomultis.loc[j,"takenDate"]==takenDate:
                BBDDaceitestotalHugomultis.loc[j,"lubName"]=lubname

   

BBDDaceitestotalHugomultis.to_excel(r'C:\Users\Usuario\Documents\Master\arreglar lo de aceites\pruebaspython\BBDDaceites-V6-ML-Finaceite.xlsx', index = False) 
#%%

# ++++++++ FIN ++++++


#++++++++++++++++++++++................................................................................................................xñN

  