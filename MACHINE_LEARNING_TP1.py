# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 02:06:04 2022

@author: 10
"""

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def pca_(df):
    
    component_v=[] #No el de Homelander
    df_v = pd.DataFrame()
    k_iter  = 4
    c_iter = 2
    
    #df = df.head(500)
    #df.to_csv(str(q)+".csv")
        
    #GENERO PCA PARA UN NUMERO CON UN MAXIMO DE 25 COMPOENTES QUE EXPLIQUEN A PARTIR DEL DATAFRAME DF
    
    for k in range(1,k_iter):
        
        n_= c_iter * k
        pca = PCA(n_components=n_)
        pca.fit_transform(df);
        
        v=pca.singular_values_
        l=[]
        sw=True
        
        for i in range(0,n_):
            
          value=sum(v[0:i])/sum(v)
          
          l.append(value)
          
          if value > .9 and sw:
            sw=False
            print("Son suficiente {} componentes para explicar los 784 atributos con un {}%".format(i,value*100))
            break
        
        index_= [i for i in range(0,n_)]
        df_componentes_pca=pd.DataFrame(pca.components_,columns=df.columns,index = index_)
        
        columnas = list(df.columns)
        datos = list(sum(np.abs(pca.components_)))
        str_="Correlación por atributo para {} componentes".format(n_)
        df_corr=pd.DataFrame(data = datos,index=columnas,columns=[str_])
        df_corr.style.set_properties(**{'text-align': 'center'})
        df_v["Componentes PCA:"+str(n_)] = df_corr[str_]
        print(df_v)
 

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

    


df = pd.read_csv("FRAUD.csv")
df_pca = df.iloc[:, :-1]
df_pca = df_pca.iloc[:, :-1]


#DESCARTAMOS EL ID DE LA CUENTA YA QUE LOS FRAUDES NO SON CONSTANES EN UNA CUENTA MAS DE UNA VEZ

df_pca = df_pca.drop(columns=["nameOrig","nameDest"])

#DESCARTAMOS EL ID DE LA CUENTA YA QUE LOS FRAUDES NO SON CONSTANES EN UNA CUENTA MAS DE UNA VEZ


dict_ = {'CASH_IN':1, 'CASH_OUT':2, 'DEBIT':3, 'PAYMENT':4, 'TRANSFER':5}


df_pca["type"] = df_pca["type"].map(dict_)
pca_(df_pca)


#DESCARTAMOS STEP POR SU BAJA PARTICIPACIÓN EN PCA Y DESCARTAMOS 
df_pca = df_pca.drop(columns = ["step","type","amount"])


x = df_pca.tail(50000)
y = df["isFraud"].tail(50000)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
y_hat = model.predict(xtest)
metrics_(y_hat, 1,0,ytest)


"""
#PREDICCIÓN

df_pca["isFraud"] = df["isFraud"]
df_pca_0 = df_pca.copy()
df_pca_1 = df_pca.copy()

df_test_1 = df_pca_1.loc[df_pca_1["isFraud"] == 1].head(500)
y_test_1 = df_test_1["isFraud"]
df_test_1 = df_test_1.iloc[:, :-1]

df_test_0 = df_pca_0.loc[df_pca_0["isFraud"] == 0].head(500)
y_test_0 = df_test_0["isFraud"]
df_test_0 = df_test_0.iloc[:, :-1]


y_hat = model.predict(df_test_1)
metrics_(y_hat, 1,0,y_test_1)

y_hat = model.predict(df_test_0)
metrics_(y_hat, 0,1,y_test_0)


#PREDICCIÓN

acum=0
for i in y_test_1:
    if i == 1:
        acum = acum+1

print(acum)
""" 


