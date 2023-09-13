#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !/usr/bin/env python
# coding: utf-8

##Data Extraction, Understanding and Visualization
##Data Preparation
##Training the Model
##Residual Analysis
##Building an understanding o important independent variables affecting target variable
##Conlcusion

# # Step1: Reading and Understanding the data

# In[1]:


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Read the data from data set to understand data
bikesharing=pd.read_csv(r'C:\Upgrad\day.csv')
bikesharing.head()


# In[6]:


bikesharing.shape


# In[8]:


bikesharing.info()


# As per the observations, it can be observed that important variables being Temparature, Season(FALL), Year month being which can directly impact the increase in target variable, which is "cnt"
