#!/usr/bin/env python
# coding: utf-8

# # Here are the 5 major steps involed in Multiple Linear regersssion
# 1. Reading, understanding and visulising the data
# 2. Prepring the data model (train-test split, rescaling etc)
# 3. Training the model
# 4. Residual Analysis
# 5. Decide which features have more impact on dependent variaable (cnt)

# # Step1: Reading and Understanding the data

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


# In[2]:


# Read the data from data set to understand data
bikesharing=pd.read_csv(r'C:\upgrad\LinearRegressionAssignment\day.csv')
bikesharing.head()


# In[3]:


bikesharing.shape


# #This data set has 730 rows and 16 columns

# In[4]:


bikesharing.info()


# #This data set does not have any missing values

# In[5]:


bikesharing.describe()


# # As per data dictionary it can be understood numberic values in respective columns as below
# ### season : season (1:spring, 2:summer, 3:fall, 4:winter)
# ### yr : year (0: 2018, 1:2019)
# ### mnth : month ( 1 to 12)
# ### holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# ### weekday : day of the week
# ### workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# ### weathersit : 
#     ### 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     ### 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     ### 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     ### 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# ### temp : temperature in Celsius
# ### atemp: feeling temperature in Celsius
# ### hum: humidity
# ### windspeed: wind speed
# ### casual: count of casual users
# ### registered: count of registered users
# ### cnt: count of total rental bikes including both casual and registered

# In[6]:


# As numeric values are already present for categorical values it will be helpful in Encoding values


# In[7]:


# Visualize only numeric variables as per data dictionary 
sns.pairplot(data=bikesharing,x_vars=['temp','atemp','hum','windspeed','weathersit','weekday','mnth','yr'],y_vars='cnt')
plt.show()


# In[8]:


# As per above charts we can understand temp, atemp, hum, windspeed are mostly following linear trend 
# but temp and atemp are more aligned with linear trend


# In[9]:


# Visualize only Categorical variables as per data dictionary 
plt.figure(figsize=(20,12))

plt.subplot(2,3,1)
sns.boxplot(x='season', y='cnt', data=bikesharing)

plt.subplot(2,3,3)
sns.boxplot(x='holiday', y='cnt', data=bikesharing)

plt.subplot(2,3,4)
sns.boxplot(x='workingday', y='cnt', data=bikesharing)


# In[10]:


# We can observe trend for all category features.
# For example season summmer and fall target variable cnt is high
# 2019 has more cnt than 2018
# Months trend is like bell curve from beginning to end of months in year
# holiday has more impact on cnt variable
# Middle day of the week and last day of the week cnt is high
# Non working day cnt is high 


# # Preparing data for modeling
#     Converting binary yes/no variables to 1 or 0
#     Other category vars to dummy vars
#     

# In[11]:


# yes/no variables or category varibles.  We should have them in the form of 1 or 0 (numeric format)
varlist=['holiday','workingday']
#bikesharing[varlist]=bikesharing[varlist].apply(lambda x:x.map({'yes':1, 'no':0}))
# We don't need above step because our data set already have in the form of 0s and 1s as per data dictionary
bikesharing[varlist].head()


# # Create dummy variables
# 

# In[12]:


#creating dummy variables for season
# dropping a redundant dummy var from each dummy variables set

dmy_season=pd.get_dummies(bikesharing['season'],drop_first=True)
dmy_season.head()



# In[13]:


# concat the dummy variables with the actual data frame
bikesharing=pd.concat([bikesharing,dmy_season],axis=1)
bikesharing.head()


# In[14]:


# Dropping unnecessary features from data frame
bikesharing=bikesharing.drop('season',axis=1)
bikesharing.head()

# casual and registered together can define cnt variable so we can delete casual and registered variables
bikesharing=bikesharing.drop('casual',axis=1)
bikesharing.head()

bikesharing=bikesharing.drop('registered',axis=1)
bikesharing.head()

# Not a numeric variable
bikesharing=bikesharing.drop('dteday',axis=1)
bikesharing.head()

# We can delete index variable
bikesharing=bikesharing.drop('instant',axis=1)
bikesharing.head()

# temp can define atemp
bikesharing=bikesharing.drop('atemp',axis=1)
bikesharing.head()


# # Split data set into train and test
# 

# In[15]:


df_train, df_test = train_test_split(bikesharing, train_size=0.7, random_state=100)
print(df_train.shape)
print(df_test.shape)


# # Resclaing the features
# 

# In[16]:


# Normalization: (x-xmin)/(xmax-xmin)
# standardiztion: (x-mu)/sigma


# In[17]:


# We will use Min Max scaling
# Instantiate object
#Create list of numeric vars

#Fit(): learns xmin, mxax
#transform(): x-xmin/xmax-xmin
# fit_transform()

scalar=MinMaxScaler()
num_vars=['temp','hum','windspeed','weathersit','mnth','weekday','yr']

df_train[num_vars]=scalar.fit_transform(df_train[num_vars])
df_train.head()



# In[18]:


df_train[num_vars].describe()
#observe min and max below between 0 and 1


# # Training the model

# In[19]:


# Heatmap
plt.figure(figsize=(16,10))
sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
plt.show()


# In[20]:


df_train.head()


# In[21]:


# X_train, y_train
y_train=df_train.pop('cnt')
X_train=df_train


# In[22]:


X_train.head()


# In[23]:


y_train.head()


# In[24]:


# Build model
# Statistical significance of variables

#add a constant
X_train_sm=sm.add_constant(X_train['temp'])

#Crete first model 
lr=sm.OLS(y_train, X_train_sm)
lr_model=lr.fit()

lr_model.params


# In[25]:


lr_model.summary()


# In[26]:


# Add another variable year

X_train_sm = X_train[['temp','yr']]
X_train_sm = sm.add_constant(X_train_sm)

#Crete  model 
lr=sm.OLS(y_train, X_train_sm)
lr_model=lr.fit()

lr_model.params

lr_model.summary()


# In[ ]:





# In[27]:


# Add another variable month

X_train_sm = X_train[['temp','yr','mnth']]
X_train_sm = sm.add_constant(X_train_sm)

#Crete  model 
lr=sm.OLS(y_train, X_train_sm)
lr_model=lr.fit()

lr_model.params

lr_model.summary()


# In[28]:


# Add another variable '3' in season

X_train_sm = X_train[['temp','yr','mnth',3]]
X_train_sm = sm.add_constant(X_train_sm)

#Crete  model 
lr=sm.OLS(y_train, X_train_sm)
lr_model=lr.fit()

lr_model.params

lr_model.summary()


# In[29]:


# Add another variable '4' in season

X_train_sm = X_train[['temp','yr','mnth',3, 2]]
X_train_sm = sm.add_constant(X_train_sm)

#Crete  model 
lr=sm.OLS(y_train, X_train_sm)
lr_model=lr.fit()

lr_model.params

lr_model.summary()


# In[30]:


# Add another variable '3' in season

X_train_sm = X_train[['temp','yr','mnth',3, 4]]
X_train_sm = sm.add_constant(X_train_sm)

#Crete  model 
lr=sm.OLS(y_train, X_train_sm)
lr_model=lr.fit()

lr_model.params

lr_model.summary()


# In[ ]:


# Observed temp, yr, mnth and season 3 (fall) has more impact on target variable "cnt" based on P value, R-Square and coefficient values

