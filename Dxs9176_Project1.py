#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 01:22:53 2019

@author: dikshasharma
"""
#Diksha Sharma
#1001679176

#importing libraries
import numpy as np
import pandas as pd
import statistics

#importing iris dataset
dataset=open("iris.data")
data=pd.read_csv(dataset,header=None,names=["X1","X2","X3","X4","Target"])

#shuffling the data
data = data.sample(frac=1)

#Dividing the source and target variables
Source_var=data.drop(["Target"],axis=1)
Source_var.insert(0,"Ones",1)
Target_var=data["Target"]

#String to Integer  mapper function
target_data= dict([(y,x+1) for x,y in enumerate(sorted(set(Target_var)))])
target_list=[]
for j in Target_var:
    for key,value in target_data.items():
        if key==j:
            target_list.append(value)
Target_df=pd.DataFrame(target_list)

#K -fold cross Validation  
def k_fold_cross_validation(folds):
    X_train_final = []
    X_test_final= []
    train = []
    Y_train_final = []
    Y_test_final= []
    Y_train_ = []
    predictions = []
    testing_set = []
    accuracy_list = []
    accuracy_perc = []
    
#Splitting the source and the target dataframes into number of folds    
    Source_split=np.split(Source_var,folds)
    Target_split=np.split(Target_df,folds)
    X_test=[]
    X_t=[]
    Y_t=[]
    Beta_values=[]
    final=[]
#using loops to split the X_test,Y_test,X_train and Y_train data     
    for i in range(len(Source_split)):
         X_train=[]
         for j in range(len(Source_split)):
             if i==j:
                 X_test.append(Source_split[j])
             else:
                 X_train.append(Source_split[j])
         X_train_final.append(X_train)
    for k in X_train_final:
        train.append(pd.concat(k))
        
    Y_test=[] 
    for l in range(len(Target_split)):
         Y_train=[]
         for m in range(len(Target_split)):
             if l==m:
                 Y_test.append(Target_split[m])
             else:
                 Y_train.append(Target_split[m])
         Y_train_final.append(Y_train)
    for k in Y_train_final:
        Y_train_.append(pd.concat(k))
    
    for each_df in (train): 
       X_t.append(np.matrix(each_df))
       
    for each_dfs in (Y_train_):
        Y_t.append(np.matrix(each_dfs))
        
 #Calculating Beta Values       
    for each_matrix in range(len( X_t)):
        first=np.dot(X_t[each_matrix].transpose(),X_t[each_matrix])
        inverse=np.linalg.inv(first)
        Beta=np.dot(inverse,X_t[each_matrix].transpose())   
        Beta_values.append(np.dot(Beta,Y_t[each_matrix]))
    for each_m in range(len(X_test)):
        final.append(np.dot(np.matrix(X_test[each_m]),np.matrix(Beta_values[each_m])))

    for i in final:
        predictions.append(np.around(i))
 
   
    accuracy_list=[]
    fold = int(len(Source_split)/folds)
    i = 0
    
    for item in range(len(Y_test)):
        testing_set.append(list(Y_test[item][i]))
        
#Calculating Accuracies
    for i in range(len(predictions)):
        count = 0
        for j in range(len(predictions[i])):
            if(predictions[i][j] == testing_set[i][j]):
                count+=1
                
        accuracy_list.append(count)
    for i in range(len(accuracy_list)):
        accuracy_perc.append(accuracy_list[i]/len(predictions[i])*100)
        
#Average of the accuracy
        mean_=str(round(statistics.mean(accuracy_perc),2))+"%"
        
    return mean_
        

print("RESULTS WITH SHUFFLING OF DATA")
print ("Accuracy for 3 fold cross validation is: {}".format(k_fold_cross_validation(3)))
print ("Accuracy for 5 fold cross validation is: {}".format(k_fold_cross_validation(5)))
print ("Accuracy for 10 fold cross validation is: {}".format(k_fold_cross_validation(10)))