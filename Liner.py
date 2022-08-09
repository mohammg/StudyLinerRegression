# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:16:57 2022

@author: moham
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def getdata(l ,variance,step=2,correlation=False):
    val=0
    ys=[]
    for item in np.arange(l):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        if correlation and correlation=='neg':
            val-=step
    xs=[ x for x in np.arange(len(ys))]
    return np.array(xs,dtype=np.float64) ,np.array(ys,dtype=np.float64)

X,y=getdata(30,1,2)
df=pd.DataFrame(zip(X,y),columns=['X','y'])
df.insert(0,'One',1)
plt.scatter(X, y)

def best_fit(xs,ys):
    
    a=0
    if ((np.mean(xs)**2)-(np.mean(xs*xs)))!=0:
        a=((np.mean(xs)*np.mean(ys))-np.mean(xs*ys))/((np.mean(xs)**2)-(np.mean(xs*xs)))
    
        
    
    b=np.mean(ys)-a*np.mean(xs)
    return a,b

yy=df['y']
xx=df[['One','X']]
a,b=best_fit(X, y)
#print(x1)

#model=LinearRegression()
#model.fit(np.array(df[['One','X']]),df['y'])
#print(model.coef_)
#print(model.intercept_)
print(str(a)+'-'+str(b))
a1,b1=best_fit2(np.array(df[['One','X']]),df['y'])
#print(str(a1)+'                 '+str(b1))
regresion_line=np.array([ (a*x)+b for x in X])
plt.plot(X, regresion_line,color='red')


def coffition(y,y_p):
    y_mean=np.mean(y)
    squred_error_p=((y_p-y)**2).sum()
    squred_error_m=((y_mean-y)**2).sum()
    n=len(y)
    #squred_error_m=0
    #squred_error_p=0
    #for i in np.arange(n):
    #    squred_error_p+=(y_p[i]-y[i])**2
    #    dif2=y_mean-y[i]
     #   dif2_s=
        
    
    re_sq=1- squred_error_p/squred_error_m
    return re_sq
cofe=coffition(y,np.array(regresion_line))

model =LinearRegression().fit(X.reshape(-1,1),y)
print(model.coef_)
print(model.intercept_)
print(cofe)
print(np.corrcoef(X,y)[0,1])



        
