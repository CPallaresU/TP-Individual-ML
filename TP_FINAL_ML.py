# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 19:03:52 2022

@author: 10
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA



def metrics_ (y_hat, num_vp, num_fp ,y_test_1):

    cont_pos = 0
    cont_neg = 0
    cont = 0
    cont_ = 0
    
    for k in range(len(y_hat)):
        
        if y_hat[k] == list(y_test_1)[k]:
            
            cont = cont + 1
            
            if  list(y_test_1)[k] == num_vp: #SI Y[K] SÍ ES FRAUDE
                
                cont_pos = cont_pos + 1 #VP
            
        else:
            
            if y_hat[k] == num_vp and  list(y_test_1)[k] == num_fp: #SI Y[K] NO ES FRAUDE
                
                cont_neg = cont_neg + 1 #FP
            
            if y_hat[k] == num_fp and  list(y_test_1)[k] == num_vp: 
            
                cont_ = cont_ + 1 #FN
    
    print("Precisión: {}".format(cont_pos/(cont_neg + cont_pos)))
    print("Exactitud: {}".format(cont/len(y_hat)))
    print("Recall: {}".format(cont_pos/(cont_ + cont_pos)))
    print("F1 SCORE : {}".format((2 * (cont_pos/(cont_neg + cont_pos)) * (cont_pos/(cont_ + cont_pos))) / ((cont_pos/(cont_neg + cont_pos)) + (cont_pos/(cont_ + cont_pos))) ))


df = pd.read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')
df["FUEL"].replace({"gasoline": 70, "kerosene": 30, "lpg" : 90, "thinner":  50}, inplace=True)
df.pop("STATUS")

df_v = pd.DataFrame()

for k in range(1,len(df.columns)):
        
        pca = PCA(n_components=k)
        pca.fit_transform(df);
        
        v=pca.singular_values_
        l=[]
        sw=True
        
        for i in range(0,k):
            
          value=sum(v[0:i])/sum(v)
          
          l.append(value)
          
          if value > .9 and sw:
            sw=False
            #print("Son suficiente {} componentes para explicar los 784 atributos con un {}%".format(i,value*100))
            break
        
        index_= [i for i in range(0,k)]
        df_componentes_pca=pd.DataFrame(pca.components_,columns=df.columns,index = index_)
        
        columnas = list(df.columns)
        datos = list(sum(np.abs(pca.components_)))
        str_="Correlación por atributo para {} componentes".format(k)
        df_corr=pd.DataFrame(data = datos,index=columnas,columns=[str_])
        df_corr.style.set_properties(**{'text-align': 'center'})
        df_v["Componentes PCA:"+str(k)] = df_corr[str_]
        #print(df_v)
        

#DISTANCE
#FREQUENCY
#FUEL

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import time

for i in range(3):
    
    df = pd.read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx')
    df["FUEL"].replace({"gasoline": .70, "kerosene": .30, "lpg" : .90, "thinner":  .50}, inplace=True)
    
    df = df.sample(frac=1,random_state=0).reset_index(drop=True)
    
    x = df.tail(50000)
    x = x.drop(columns  = "STATUS")
    print("")
    print("")

    if i == 0:
        t = "FUEL","DISTANCE"
        print(t)
        x = x[["FUEL","DISTANCE"]]
        
        
    else:
        
        if i == 1 :
            t = "DESIBEL","AIRFLOW","FUEL"
            print(t)
            x = x[["DESIBEL","AIRFLOW","FUEL"]]
            
        else:
            
            #x = x.drop(columns =  "DISTANCE")
            print("Todas las columnas")
        
    y = df["STATUS"].tail(50000)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
    model = DecisionTreeClassifier()
    start_time = time.time()
    model.fit(xtrain, ytrain)
    print("--- %s seconds ---" % (time.time() - start_time))
    y_hat = model.predict(xtest)
    metrics_(y_hat, 1,0,ytest)








