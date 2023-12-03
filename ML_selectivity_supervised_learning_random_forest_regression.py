#import all the requIred libraries##
import os
import sys
from functools import partial

import ase
import numpy as np
import pandas as pd
from ase.visualize import view
import xlrd
from ase.io import read, write
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from sklearn.inspection import permutation_importance

from itertools import combinations



def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(2)
   tf.random.set_seed(2)
   np.random.seed(2) 

vali = "ch3ch3_au(100)_secNN"

##Inputs from ads-ads paper##
dataset = pd.read_excel("input_data.xlsx",sheet_name=vali)
X = dataset.drop(columns=['energy'])
#print(X)
Y = dataset['energy']
#print(Y)

#Separate target variable and predictor variables##
targetvariable = 'energy'
predictors = ['a1','a2','a3','a4','a11','a22','a33','a44']
X = dataset[predictors].values
y = dataset[targetvariable].values

dataset_vali = pd.read_excel("validation_data.xlsx",sheet_name=vali)
X_vali = dataset_vali.drop(columns=['energy'])
#print(X)
Y_vali = dataset_vali['energy']
#print(Y)
X_vali = dataset_vali[predictors].values
y_vali = dataset_vali[targetvariable].values

#adsads = 'Pt(100)_CCH3CCH3_second_nearest'

##plotting y's for text##
p1 = 2.50
p2 = 2.30
p3 = 2.10

#Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

###### Random Forest Regression in Python #######
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

#Train everything together#
# RegModel = RandomForestRegressor(n_estimators=200,criterion='squared_error')
# RF=RegModel.fit(X_train,y_train)
# prediction=RF.predict(X_test)
# predictions_all = np.array([tree.predict(X) for tree in RegModel.estimators_])


#Split the data into training and testing set
X = dataset[predictors].values
y = dataset[targetvariable].values
kf = KFold(n_splits=3)
#print(kf)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

Input_Shape = [X.shape[1]]

RF_pred = [None]*len(y)
Train_RF_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    RegModel = RandomForestRegressor(n_estimators=200,criterion='squared_error')
    RF=RegModel.fit(X_train,y_train)
    pred_values = RF.predict(X_test)
    train_pred_values = RF.predict(X_train)
    
    for i,v in enumerate(pred_values):
        RF_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_RF_pred[(i + count*int(len(y)/3))] = v
    
 #   RF_DF = pd.DataFrame(RF.history)

 #   RF_DF.loc[:, ['loss','val_loss']].plot()
 #   RF_DF.savefig('./val_loss_random_forest.png')
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(RF_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    #print(count)
    train_score = mean_absolute_error(Train_RF_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,RF_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_RF_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, RF.predict(X)))
# print('R2 Value:',np.mean(Train_r2))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - RF.predict(X)))) * 100))
# print('Accuracy',100- ((np.mean(CV_scores))*100))

# #Plotting the feature importance for Top 10 most important columns

feature_importances = pd.Series(RF.feature_importances_, index=predictors)
plot1 = feature_importances.nlargest(20).plot(kind='barh')
fig1 = plot1.get_figure()
#fig1.savefig(f"./feature_importances_{adsads}_RF_split.png")

fig,ax = plt.subplots(1)
ax.scatter(y,RF.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(RF)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, RF.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - RF.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - RF.predict(X)))), fontsize=14)
#plt.text(-0.5, -1.2, 'MAE for blind set (eV): {:.2f}'.format(np.mean(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
#plt.text(-0.5, -1.1, 'MAX for blind set (eV): {:.2f}'.format(np.max(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
plt.draw()
#plt.savefig(f'./actual_vs_pred_{adsads}_RF_split')



#Printing some sample values of prediction

print('R2 Value of blind set RF:',metrics.r2_score(y_vali, RF.predict(X_vali)))
print('MAE of blind set RF:',np.mean(np.abs(y_vali - RF.predict(X_vali))))
print('MAX of blind set RF:',np.max(np.abs(y_vali - RF.predict(X_vali))))

# lax = [0,7]
# lax = np.array(lax).reshape((-1,1))
# prediction_test=RF.predict(lax)
# print(prediction_test)



######## Kernel Ridge regression #######
from sklearn.kernel_ridge import KernelRidge

#Separate target variable and predictor variables##

X = dataset[predictors].values
y = dataset[targetvariable].values

kf = KFold(n_splits=3)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

kr_pred = [None]*len(y)
Train_kr_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []


for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    krr = KernelRidge(alpha=1.0)
    KR=krr.fit(X_train,y_train)
    pred_values = KR.predict(X_test)
    train_pred_values = KR.predict(X_train)
    
    for i,v in enumerate(pred_values):
        kr_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_kr_pred[(i + count*int(len(y)/3))] = v
    
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(kr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    #print(count)
    train_score = mean_absolute_error(Train_kr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,kr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_kr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, KR.predict(X)))


#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - KR.predict(X)))) * 100))

fig,ax = plt.subplots(1)
ax.scatter(y,KR.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(KR)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, KR.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - KR.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - KR.predict(X)))), fontsize=14)

plt.draw()
#plt.savefig(f'./kernel_ridge_prediction/actual_vs_pred_{adsads}_KR_split')



#Printing some sample values of prediction

print('R2 Value of blind set KR:',metrics.r2_score(y_vali, KR.predict(X_vali)))
print('MAE of blind set KR:',np.mean(np.abs(y_vali - KR.predict(X_vali))))
print('MAX of blind set KR:',np.max(np.abs(y_vali - KR.predict(X_vali))))


######## Gaussian regression #######
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


#Separate target variable and predictor variables##

X = dataset[predictors].values
y = dataset[targetvariable].values

kf = KFold(n_splits=3)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

gpr_pred = [None]*len(y)
Train_gpr_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []


for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)
    GPR=gpr.fit(X_train,y_train)
    pred_values = GPR.predict(X_test)
    train_pred_values = GPR.predict(X_train)
    
    for i,v in enumerate(pred_values):
        gpr_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_gpr_pred[(i + count*int(len(y)/3))] = v
    
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(gpr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    #print(count)
    train_score = mean_absolute_error(Train_gpr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,gpr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_gpr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, GPR.predict(X)))


#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - GPR.predict(X)))) * 100))


fig,ax = plt.subplots(1)
ax.scatter(y,GPR.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(GPR)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, GPR.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - GPR.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - GPR.predict(X)))), fontsize=14)

plt.draw()
#plt.savefig(f'./gaussian_process_prediction/actual_vs_pred_{adsads}_GPR_split')



#Printing some sample values of prediction

print('R2 Value of blind set GPR:',metrics.r2_score(y_vali, GPR.predict(X_vali)))
print('MAE of blind set GPR:',np.mean(np.abs(y_vali - GPR.predict(X_vali))))
print('MAX of blind set GPR:',np.max(np.abs(y_vali - GPR.predict(X_vali))))


######## Bayesian Ridge regression #######
from sklearn import linear_model

#Separate target variable and predictor variables##

X = dataset[predictors].values
y = dataset[targetvariable].values

kf = KFold(n_splits=3)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

brr_pred = [None]*len(y)
Train_brr_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []


for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    kernel = DotProduct() + WhiteKernel()
    brr = linear_model.BayesianRidge()
    BRR=brr.fit(X_train,y_train)
    pred_values = BRR.predict(X_test)
    train_pred_values = BRR.predict(X_train)
    
    for i,v in enumerate(pred_values):
        brr_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_brr_pred[(i + count*int(len(y)/3))] = v
    
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(brr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    #print(count)
    train_score = mean_absolute_error(Train_brr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,brr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_brr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, BRR.predict(X)))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - BRR.predict(X)))) * 100))

fig,ax = plt.subplots(1)
ax.scatter(y,BRR.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(BRR)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, BRR.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - BRR.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - BRR.predict(X)))), fontsize=14)
plt.draw()
#plt.savefig(f'./bayesian_ridge_prediction/actual_vs_pred_{adsads}_BRR_split')



#Printing some sample values of prediction

print('R2 Value of blind set BRR:',metrics.r2_score(y_vali, BRR.predict(X_vali)))
print('MAE of blind set BRR:',np.mean(np.abs(y_vali - BRR.predict(X_vali))))
print('MAX of blind set BRR:',np.max(np.abs(y_vali - BRR.predict(X_vali))))


######## LASSO regression #######
from sklearn.linear_model import Lasso

#Separate target variable and predictor variables##

X = dataset[predictors].values
y = dataset[targetvariable].values

kf = KFold(n_splits=3)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

lasso_pred = [None]*len(y)
Train_lasso_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []


for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    lasso = Lasso()
    LASSO=lasso.fit(X_train,y_train)
    pred_values = LASSO.predict(X_test)
    train_pred_values = LASSO.predict(X_train)
    
    for i,v in enumerate(pred_values):
        lasso_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_lasso_pred[(i + count*int(len(y)/3))] = v
    
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(lasso_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    #print(count)
    train_score = mean_absolute_error(Train_lasso_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    test_r2_score = r2_score(y_test,lasso_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_lasso_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, LASSO.predict(X)))
# print('R2 Value:',np.mean(Train_r2))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - LASSO.predict(X)))) * 100))

fig,ax = plt.subplots(1)
ax.scatter(y,LASSO.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(LASSO)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, LASSO.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - LASSO.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - LASSO.predict(X)))), fontsize=14)

plt.draw()
#plt.savefig(f'./lasso_prediction/actual_vs_pred_{adsads}_lasso_split')

#Printing some sample values of prediction

print('R2 Value of blind set LASSO:',metrics.r2_score(y_vali, LASSO.predict(X_vali)))
print('MAE of blind set LASSO:',np.mean(np.abs(y_vali - LASSO.predict(X_vali))))
print('MAX of blind set LASSO:',np.max(np.abs(y_vali - LASSO.predict(X_vali))))

######## Linear regression #######
from sklearn.linear_model import LinearRegression

#Separate target variable and predictor variables##

X = dataset[predictors].values
y = dataset[targetvariable].values

kf = KFold(n_splits=3)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

lr_pred = [None]*len(y)
Train_lr_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []


for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    lr = LinearRegression()
    LR=lr.fit(X_train,y_train)
    pred_values = LR.predict(X_test)
    train_pred_values = LR.predict(X_train)
    
    for i,v in enumerate(pred_values):
        lr_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_lr_pred[(i + count*int(len(y)/3))] = v
    
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(lr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    #print(count)
    train_score = mean_absolute_error(Train_lr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,lr_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_lr_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, LR.predict(X)))
# print('R2 Value:',np.mean(Train_r2))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs((y - LR.predict(X)))) * 100))
# print('Accuracy',100- ((np.mean(CV_scores))*100))

fig,ax = plt.subplots(1)
ax.scatter(y,LR.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(LR)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, LR.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - LR.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - LR.predict(X)))), fontsize=14)
plt.draw()
#plt.savefig(f'./linear_prediction/actual_vs_pred_{adsads}_lr_split')

#Printing some sample values of prediction

print('R2 Value of blind set LR:',metrics.r2_score(y_vali, LR.predict(X_vali)))
print('MAE of blind set LR:',np.mean(np.abs(y_vali - LR.predict(X_vali))))
print('MAX of blind set LR:',np.max(np.abs(y_vali - LR.predict(X_vali))))

####### Neural Network algorithm #######

##Inputs from ads-ads paper##

#Separate target variable and predictor variables##

X = dataset[predictors].values
y = dataset[targetvariable].values

kf = KFold(n_splits=3)
reset_random_seeds()
Input_Shape = [X.shape[1]]

# standardize dataset

NN_pred = [None]*len(y)
Train_NN_pred = [None]*len(y)*3

count = 0
CV_scores = []
Train_CV_scores = []

Test_r2 = []
Train_r2 = []

for train_index , test_index in kf.split(X):
    
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #print(train_index)
    
    reset_random_seeds()
    
    NN_model = keras.Sequential([
    layers.BatchNormalization(input_shape = Input_Shape),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(128, activation = 'sigmoid'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(64, activation = 'sigmoid'),
    layers.Dense(1)
    ])

    NN_model.compile(optimizer = 'adam', loss = 'mae')
    
    history = NN_model.fit(X_train,y_train, batch_size = 100, epochs = 600, validation_split = 0.2, verbose = 0)
    pred_values = NN_model.predict(X_test)
    train_pred_values = NN_model.predict(X_train)
    
    for i,v in enumerate(pred_values):
        NN_pred[(i + count*int(len(y)/3))] = v
        
    for i,v in enumerate(train_pred_values):
        Train_NN_pred[(i + count*int(len(y)/3))] = v
    
    history_DF = pd.DataFrame(history.history)

    history_DF.loc[:, ['loss','val_loss']].plot()
    
    'Mean Absolute Error and R2 score calculation'
    
    test_score = mean_absolute_error(NN_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)],y_test)
    CV_scores.append(test_score)
    
    train_score = mean_absolute_error(Train_NN_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)],y_train)
    Train_CV_scores.append(train_score)
    
    
    test_r2_score = r2_score(y_test,NN_pred[(0+int(len(y)/3)*count):(int(len(y)/3) + int(len(y)/3)*count)])
    Test_r2.append(test_r2_score)
    
    train_r2_score = r2_score(y_train,Train_NN_pred[(0+int(len(y)/3)*count):(2*int(len(y)/3) + int(len(y)/3)*count)])
    Train_r2.append(train_r2_score)
    #print(count)
    count += 1

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y, NN_model.predict(X)))

#Measuring accuracy on Training Data
print('Accuracy',100- (np.mean(np.abs(y - NN_model.predict(X)))*100))


fig,ax = plt.subplots(1)
ax.scatter(y,NN_model.predict(X))
ax.plot([np.min(y)-0.5,np.max(y)+0.5],[np.min(y)-0.5,np.max(y)+0.5])
ax.set_xlabel('Actual (eV)')
ax.set_ylabel('Predicted (eV)')
ax.set_xlim([np.min(y)-0.5,np.max(y)+0.5])
ax.set_ylim([np.min(y)-0.5,np.max(y)+0.5])
ax1 = plt.gca()
ax1.set_aspect('equal', adjustable='box')
plt.title('{}'.format(str(NN_model)), fontsize=16)
plt.text(-0.1, p1, 'R^2: {:.2f}'.format(metrics.r2_score(y, NN_model.predict(X))), fontsize=14)
plt.text(-0.1, p2, 'MAE (eV): {:.2f}'.format(np.mean(np.abs(y - NN_model.predict(X)))), fontsize=14)
plt.text(-0.1, p3, 'MAX (eV): {:.2f}'.format(np.max(np.abs(y - NN_model.predict(X)))), fontsize=14)

#plt.text(-0.5, -1.2, 'MAE for blind set (eV): {:.2f}'.format(np.mean(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
#plt.text(-0.5, -1.1, 'MAX for blind set (eV): {:.2f}'.format(np.max(np.abs(y_test - RF.predict(X_test)))), fontsize=14)
plt.draw()
#plt.savefig(f'./neural_network_prediction/actual_vs_pred_{adsads}_NN')


#Printing some sample values of prediction

print('R2 Value of blind set NN:',metrics.r2_score(y_vali, NN_model.predict(X_vali)))
print('MAE of blind set NN:',np.mean(np.abs(y_vali - NN_model.predict(X_vali))))
print('MAX of blind set NN:',np.max(np.abs(y_vali - NN_model.predict(X_vali))))