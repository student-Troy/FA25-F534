# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 10:19:53 2025

@author: troya
"""
# Clean up
%reset -f
%clear

# Importing necessary packages
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error as mse

import os

# Assume current working directory is /home/user/project
print(f"Initial CWD: {os.getcwd()}")

# Change to a subdirectory named 'data'
os.chdir("Lecture 4 - More Regression")
print(f"CWD after entering 'data': {os.getcwd()}")

# Reading from a CSV File 
# Both features and target have already been scaled: mean = 0; SD = 1
data = pd.read_csv('Houseprice_data_scaled.csv') 


# Change back to FA25-F534 directory
os.chdir("..")
print(f"CWD after entering 'data': {os.getcwd()}")

# First 1800 data items are training set; the next 600 are the validation set
train = data.iloc[:1800] 
val = data.iloc[1800:2400]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train, X_val = train.drop('Sale Price', axis=1), val.drop('Sale Price', axis=1)
y_train, y_val = train[['Sale Price']], val[['Sale Price']] 

# Importing models
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

# Create dataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs
