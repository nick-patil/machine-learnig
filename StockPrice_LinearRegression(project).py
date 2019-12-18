# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:56:55 2019

@author: Aniket Kambli
"""

 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

def getdata(stockname,start_date,end_date):
    data=pdr.data.get_data_yahoo(stockname,start_date,end_date)
    return data
          
    
def split_data(data):
    data['Moving Average'] = data['Close'].rolling(window=20).mean()
    data['Open-Close'] = data['Open'] - data['Close'].shift(1)
    data['Open-Open'] = data['Open'] - data['Open'].shift(1)
    data=data.dropna() 
    length_of_data=data.shape[0]
    length70=int((length_of_data/100)*70)
    X=data[:length_of_data]
    y=data[:length_of_data]
    X=pd.DataFrame(X[['Moving Average','Open-Open','Open-Close']])
    y=pd.DataFrame(y['Close'])
    X_train=X.iloc[:length70]
    X_test=X.iloc[length70:]
    y_train=y.iloc[:length70]
    y_test=y.iloc[length70:]
    return X_train,X_test,y_train,y_test
    
    
def model_linear_regression(X_train,X_test,y_train,y_test):
    from sklearn.preprocessing import StandardScaler
    scx=StandardScaler()
    X_train=scx.fit_transform(X_train)
    X_test=scx.transform(X_test)
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(X_train,y_train)
    ypreds=model.predict(X_test)
    print(r2_score(y_test,ypreds))
    print(np.sqrt(mean_squared_error(y_test,ypreds)))
    y_test['predictions']=0
    y_test['predictions']=ypreds
    plt.figure(figsize=(10,5))
    plt.title("X")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(y_train['Close'],'b',label='Training data')
    plt.plot(y_test['Close'],'g',label='Testing data')
    plt.plot(y_test['predictions'],'r',label='predicted values')
    plt.legend(loc='best')
    

            
            
#main program    
data=getdata('TATASTEEL.Ns','2015-01-01','2019-01-01')
X_train,X_test,y_train,y_test=split_data(data)
model_linear_regression(X_train,X_test,y_train,y_test)
    

