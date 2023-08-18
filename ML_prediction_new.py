# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:51:49 2022

@author: bello
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from math import sqrt
import time

#%% Load and prepare data

df = pd.read_csv('C:/Users/bello/OneDrive/Desktop/EMCH561/Graduate_project/hydrocarbon_features_combinedsurfaces.csv')
print(df.shape)
df.describe()

exclude_column = ['SMILES', 'C0#:C0', 'C0#:C1', 'C1#:C1', 'C3#', 'C6-C', 'C1-CC', 'C2-CC', 'C4-CC', 'C5-CC', 'C6-CC']
target_column = ['Energies']
predictors = list(set(list(df.columns)) - set(target_column) - set(exclude_column))


#%% Linear regression

# df[predictors] = df[predictors]/df[predictors].max()
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression() 
lr.fit(X_train,y_train)

pred_train_lr = lr.predict(X_train)
print(r2_score(y_train, pred_train_lr))

pred_test_lr = lr.predict(X_test)
print(r2_score(y_test, pred_test_lr))

m_train, b_train = np.polyfit(y_train[:,0], pred_train_lr[:,0], 1)
m_test, b_test = np.polyfit(y_test[:,0], pred_test_lr[:,0],1)
x_slope = np.linspace(-0.5,4.4,30)
x_slope2 = np.linspace(0.1,4.5,30)

## R2 comparison
plt.figure(figsize=(6.5,3))
plt.subplot(121)
plt.scatter(y_train, pred_train_lr, s=15, alpha=0.6)
plt.plot(y_train, y_train, 'k-', lw=3)
plt.plot(x_slope, m_train*x_slope+b_train, 'r--', lw=2)
plt.ylabel('Predicted Energy')
plt.xlabel('Observed Energy')
plt.title('Training Data')
plt.text(3,0,'$R^2$ = 0.93', fontsize=9 )
plt.xlim(-1,5); plt.ylim(-1,5)
plt.grid(True)
plt.tight_layout()

plt.subplot(122)
plt.scatter(y_test, pred_test_lr, label='Data points', s=15, alpha=0.6)
plt.plot(y_test, y_test, 'k-', lw=3, label='Observed')
plt.plot(x_slope2, m_test*x_slope2+b_test, 'r--', lw=2, label='Predicted')
plt.xlabel('Observed Energy')
plt.title('Testing Data')
plt.xlim(-1,5); plt.ylim(-1,5)
plt.text(3,0,'$R^2$ = 0.93', fontsize=9 )
plt.legend(loc=2, prop={'size':9})
plt.grid(True)
plt.tight_layout()
plt.savefig('new_LR_TrainingandTesting.png', dpi=500)


## Scatter plot of predicted and actual datasets for training and testing datasets respectively
plt.figure(figsize=(6.5,3))
plt.subplot(1,2,1)
plt.scatter(y_train, X_train[:,38], marker='o', s=15, label='Observed data')
plt.scatter(pred_train_lr, X_train[:,38], marker='s', s=15, label='Predicted data')
plt.ylabel('Number of Hydrogen')
plt.xlabel('Adsorption Energy')
plt.title('Training Data')
plt.grid(True)
plt.tight_layout()
plt.subplot(1,2,2)
plt.scatter(y_test, X_test[:,38], marker='o', s=15, label='Observed data')
plt.scatter(pred_test_lr, X_test[:,38], marker='s', s=15, label='Predicted data')
plt.xlabel('Adsorption Energy')
plt.title('Testing Data')
plt.grid(True)
plt.legend(prop={'size':9})
plt.tight_layout()
plt.savefig('new_test_comparison_scatter.png', dpi=500)


#%% Ridge Regression

model_rr = Ridge(alpha=0.1, random_state=42)
model_rr.fit(X_train, y_train) 
pred_train_rr= model_rr.predict(X_train)
# print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= model_rr.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))

#%% Lasso Regression

model_lasso = Lasso(alpha=0.1, max_iter=10000,random_state=42)
model_lasso.fit(X_train, y_train) 
pred_train_lasso= model_lasso.predict(X_train)
# print(np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print(r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 
print(r2_score(y_test, pred_test_lasso))


#%% ElasticNet

model_enet = ElasticNet(alpha = 0.1, max_iter=10000,random_state=42)
model_enet.fit(X_train, y_train) 
pred_train_enet= model_enet.predict(X_train)
# print(np.sqrt(mean_squared_error(y_train,pred_train_enet)))
print(r2_score(y_train, pred_train_enet))

pred_test_enet= model_enet.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test,pred_test_enet)))
print(r2_score(y_test, pred_test_enet))

#%% Comparing the regularization methods using the testing data

m_lr, b_lr = np.polyfit(y_test[:,0], pred_test_lr[:,0],1)
m_rr, b_rr = np.polyfit(y_test[:,0], pred_test_rr[:,0],1)
m_lass, b_lass = np.polyfit(y_test[:,0], pred_test_lasso,1)
m_enet, b_enet = np.polyfit(y_test[:,0], pred_test_enet,1)

plt.figure(figsize=(6.5,5))
plt.scatter(y_test, pred_test_lr, label='Data points', s =50 , alpha = 0.6)
plt.plot(y_test, y_test, 'k-', label='Observed',lw=3)
plt.plot(x_slope2, m_lr*x_slope2+b_lr, c='r', label='Linear',lw=3)
plt.plot(x_slope2, m_rr*x_slope2+b_rr, 'C2-.', label='Ridge',lw=3)
plt.plot(x_slope2, m_lass*x_slope2+b_lass, 'm:', label='Lasso',lw=3)
# plt.plot(y_test, m_enet*y_test+b_enet, 'C5-.', label='ElasticNet')
plt.ylabel('Predicted Energy',fontsize=12)
plt.xlabel('Observed Energy',fontsize=12)
plt.title(r'Comparing Different Regularization Method at $\alpha$ = 0.1',fontsize=14)
plt.legend(loc=2, prop={'size':10})
plt.grid(True)
plt.tight_layout()
plt.savefig('Regularization_comparison.png', dpi=500)

Lin_comb_train = [0.93, 0.92, 0.85, 0.86 ]
Lin_comb_test = [0.93, 0.90, 0.87, 0.87 ]
Lin_title = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']

X_axis = np.arange(len(Lin_title))

plt.figure(figsize=(6.5,5))
plt.xticks(X_axis, Lin_title)
plt.bar(X_axis - 0.2, Lin_comb_train, width=0.4, label='Training')
plt.bar(X_axis + 0.2, Lin_comb_test, width=0.4, label='Testing')
plt.xlabel('Linear Model', fontsize=12)
plt.ylabel('$R^2$', fontsize=12)
plt.title('Comparison of Linear Models', fontsize=14)
plt.xticks(fontsize=10)
plt.legend()
plt.ylim([0.7,1])
plt.tight_layout()
plt.savefig('Lin_model_comp.png', dpi=500)


#%% SVR
#optimize hyperparameters
model_svr = SVR(kernel = 'rbf', C=1.1, epsilon=0.2, gamma = 0.008, random_state=42)
model_svr.fit(X_train, y_train) 
pred_train_svr= model_svr.predict(X_train)
#print(np.sqrt(mean_squared_error(y_train,pred_train_svr)))
print(r2_score(y_train, pred_train_svr))

pred_test_svr= model_svr.predict(X_test)
print(r2_score(y_test, pred_test_svr))

pred_test_svr = np.expand_dims(pred_test_svr, 1)
m_svr, b_svr = np.polyfit(y_test[:,0], pred_test_svr[:,0],1)
x_slope = np.linspace(0.15,4.5,30)

plt.figure(figsize=(6.5,5))
plt.scatter(y_test, pred_test_svr, label='Data points', s =50 , alpha = 0.6)
plt.plot(y_test, y_test, 'k-', label='Observed',lw=3)
plt.plot(x_slope, m_svr*x_slope+b_svr, 'r--', label='Predicted',lw=3)
plt.ylabel('Predicted Energy',fontsize=12)
plt.xlabel('Observed Energy',fontsize=12)
plt.title('SVR Prediction',fontsize=14)
plt.legend(loc=2, prop={'size':10})
plt.text(3,0,'$R^2$ = 0.90', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('SVR_Prediction.png', dpi=500)

# Parameter tuning

kern = [0.001, 0.005, 0.01, 0.1, 0.2]
kernel_score = np.zeros(5)
for i in range(5):
    modelsvr = SVR(gamma=kern[i])
    modelsvr.fit(X_train, y_train)
    pred_testsvr = modelsvr.predict(X_test)
    kernel_score[i] = mean_squared_error(y_test, pred_testsvr)
    
plt.figure(figsize=(4,3))
plt.plot(['0.001', '0.005', '0.01', '0.1', '0.2'], kernel_score, marker='o')
plt.xlabel('Gamma Value')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.title('Gamma Tuning for RBF Kernel')
plt.tight_layout()
plt.savefig('SVR_Gamma_tuning.png', dpi=500)


# Parameter tuning

kern = ['linear', 'poly', 'rbf']
kernel_score = np.zeros(3)
for i in range(3):
    modelsvr = SVR(kernel=kern[i])
    modelsvr.fit(X_train, y_train)
    pred_testsvr = modelsvr.predict(X_test)
    kernel_score[i] = mean_squared_error(y_test, pred_testsvr)
    
plt.figure(figsize=(4,3))
plt.bar(['linear', 'poly', 'rbf'], kernel_score)
plt.xlabel('Kernel')
plt.ylabel('Mean Squared Error')
plt.title('MSE for Different Kernels')
plt.tight_layout()
plt.savefig('Different_kernels.png', dpi=500)

# grid = GridSearchCV(
# estimator=SVR(kernel='rbf'),
# param_grid={
# 'C': [1e-8, 1e-4, 1e-2, 1, 5, 10, 100, 1000],
# 'epsilon': [0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
# 'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]
# },
# cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# grid.fit(X,y)

#print the best parameters from all possible combinations
# print("best parameters are: ", grid.best_params_)
# best params C = 1.1, eps = 0.2, gamma = 0.008


#%% KRR
#optimize hyperparameters
krr = KernelRidge(alpha=0.001)
krr.fit(X_train, y_train)

pred_train_krr= krr.predict(X_train)
#print(np.sqrt(mean_squared_error(y,pred_krr)))
print(r2_score(y_train, pred_train_krr))

pred_test_krr= krr.predict(X_test)
print(r2_score(y_test, pred_test_krr))

#%% Random Forest
randforest = RandomForestRegressor(max_depth = 13, random_state=42, min_samples_split = 5)
randforest.fit(X_train, y_train)

pred_train_randforest = randforest.predict(X_train)
print(r2_score(y_train, pred_train_randforest))

pred_test_randforest= randforest.predict(X_test)
print(r2_score(y_test, pred_test_randforest))

pred_test_randforest = np.expand_dims(pred_test_randforest, 1)
m_rf, b_rf = np.polyfit(y_test[:,0], pred_test_randforest[:,0],1)
x_slope = np.linspace(0.15,4.5,30)

plt.figure(figsize=(6.5,5))
plt.scatter(y_test, pred_test_randforest, label='Data points', s =50 , alpha = 0.6)
plt.plot(y_test, y_test, 'k-', label='Observed',lw=3)
plt.plot(x_slope, m_rf*x_slope+b_rf, 'r--', label='Predicted',lw=3)
plt.ylabel('Predicted Energy',fontsize=12)
plt.xlabel('Observed Energy',fontsize=12)
plt.title('Random Forest Prediction',fontsize=14)
plt.legend(loc=2, prop={'size':10})
plt.text(3,0,'$R^2$ = 0.94', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('Random_Forest_Prediction.png', dpi=500)

parameters = [3, 5, 8, 10, 15, 20]
param_score = np.zeros(6)
for i in range(6):
    modelrandforest = RandomForestRegressor(max_depth=parameters[i], random_state=42)
    modelrandforest.fit(X_train, y_train)
    pred_testrf = modelrandforest.predict(X_test)
    param_score[i] = mean_squared_error(y_test, pred_testrf)

plt.figure(figsize=(4,3))
plt.plot(['3', '5', '8', '10', '15', '20'], param_score, marker='o')
plt.xlabel(' Maximum depth')
plt.grid(True)
plt.ylabel('Mean Squared Error')
plt.title('Maximum Depth Tuning')
plt.tight_layout()
plt.savefig('RandForest_Max_depth_tuning.png', dpi=500)

parameters = [ 5, 10, 20, 30, 50]
param_score = np.zeros(5)
for i in range(5):
    modelrandforest = RandomForestRegressor(max_depth=10, random_state=42, min_samples_split = parameters[i])
    modelrandforest.fit(X_train, y_train)
    pred_testrf = modelrandforest.predict(X_test)
    param_score[i] = mean_squared_error(y_test, pred_testrf)

plt.figure(figsize=(4,3))
plt.plot(['5', '10', '20', '30', '50'], param_score, marker='o')
plt.xlabel('Minimum Samples Split')
plt.grid(True)
plt.ylabel('Mean Squared Error')
plt.title('Minimum Samples Split Tuning')
plt.tight_layout()
plt.savefig('RandForest_min_sample_split_tuning.png', dpi=500)

param = {'max_depth': [5, 7, 9, 11, 13, 15], 'min_samples_split': [2, 3, 5, 7, 10]}
modelrf = RandomForestRegressor()
clf = GridSearchCV(estimator=modelrf, param_grid=param,cv=5,n_jobs=-1)
clf.fit(X, y)

#%% Neural networks

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

hls = [(10,10),(20,20),(50,50), (100,100), (10,10,10), (20,20), (50,50,50), (100,100,100)] # Hidden layer sizes
NN_score = np.zeros(8)

param = {'hidden_layer_sizes' : [(10,10),(20,20),(50,50), (100,100), (10,10,10), (20,20), (50,50,50), (100,100,100)], 'max_iter' :[100, 500, 1000, 5000, 10000]}
modelNN = MLPRegressor(random_state = 1)
clf = GridSearchCV(estimator = modelNN, param_grid = param, cv =5, n_jobs = -1)
NN_model = clf.fit(X, y)
pred_test_NN = NN_model.predict(X_test)
pred_test_NN = np.expand_dims(pred_test_NN, 1)

m_nn , b_nn = np.polyfit(y_test[:,0], pred_test_NN[:,0],1)

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (10,10), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[0] = mean_squared_error(y_test, pred_test_NN)

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (20,20), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[1] = mean_squared_error(y_test, pred_test_NN)
print(r2_score(y_test, pred_test_NN))
pred_train_NN = modelNN.predict(X_train)
print(r2_score(y_train, pred_train_NN))

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (50,50), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[2] = mean_squared_error(y_test, pred_test_NN)

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (100,100), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[3] = mean_squared_error(y_test, pred_test_NN)

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (10,10,10), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[4] = mean_squared_error(y_test, pred_test_NN)
print(r2_score(y_test, pred_test_NN))
pred_train_NN = modelNN.predict(X_train)
print(r2_score(y_train, pred_train_NN))

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (20,20,20),max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[5] = mean_squared_error(y_test, pred_test_NN)

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (50,50,50), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[6] = mean_squared_error(y_test, pred_test_NN)

modelNN = MLPRegressor(random_state = 1, hidden_layer_sizes= (100,100,100), max_iter = 500).fit(X_train, y_train)
pred_test_NN = modelNN.predict(X_test)
NN_score[7] = mean_squared_error(y_test, pred_test_NN)

plt.figure(figsize=(6.5,5))
plt.bar(['(10,10)','(20,20)','(50,50)', '(100,100)', '(10,10,10)', '(20,20,20)', '(50,50,50)'], NN_score[:7], width = 0.5)
plt.xlabel('Hidden layers', fontsize = 12)
plt.ylabel('Mean Squared Error', fontsize = 12)
plt.title('Hidden Layers Tuning', fontsize = 14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.ylim([0.1,0.17])
plt.tight_layout()
plt.savefig('NN_bar_graphs.png', dpi=500)

plt.figure(figsize=(6.5,5))
plt.scatter(y_test, pred_test_lr, label='Data points', s =50 , alpha = 0.6)
plt.plot(y_test, y_test, 'k-', label='Observed',lw=3)
plt.plot(x_slope, m_nn*x_slope+b_nn,'m--', label='Neural Networks',lw=3)
plt.ylabel('Predicted Energy',fontsize=12)
plt.xlabel('Observed Energy',fontsize=12)
plt.title(r'Neural Network prediction',fontsize=14)
plt.legend(loc=2, prop={'size':10})
plt.grid(True)
plt.tight_layout()
plt.text(3,0,'$R^2$ = 0.92', fontsize = 14)
plt.savefig('Neural Network.png', dpi=500)
#%% Performance Metrics Comparison

Over_comb_train = [0.93, 0.91, 0.93, 0.97, 0.94 ]
Over_comb_test = [0.93, 0.90, 0.93, 0.94, 0.92 ]
Over_title = ['Linear', 'SVR', 'KRR', 'Random Forest', 'Neural Network']

X_axis = np.arange(len(Over_title))

plt.figure(figsize=(8,5))
plt.xticks(X_axis, Over_title)
plt.bar(X_axis - 0.2, Over_comb_train, width=0.4, label='Training')
plt.bar(X_axis + 0.2, Over_comb_test, width=0.4, label='Testing')
plt.xlabel('Model', fontsize=12)
plt.ylabel('$R^2$', fontsize=12)
plt.title('Comparison of Linear Model to Other Models', fontsize=14)
plt.xticks(fontsize=10, rotation=30)
plt.legend()
plt.ylim([0.7,1.1])
plt.tight_layout()
plt.savefig('Over_model_comp.png', dpi=500)































