#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import data from csv file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('prices-split-adjusted.csv')


# In[2]:


print(data.head())
#shape of data
data.shape


# In[3]:


#convert date to numerical value
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].dt.strftime('%Y%m%d').astype(int)


# In[4]:


data.head()


# In[5]:


#encode stock symbol
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['symbol'] = le.fit_transform(data['symbol'])


# In[6]:


data.head()
data.dtypes


# In[7]:


data['date'] = data['date'].astype('float64')
data['symbol'] = data['symbol'].astype('float64')
#standardize data except date and symbol
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['open', 'close', 'low', 'high', 'volume']] = scaler.fit_transform(data[['open', 'close', 'low', 'high', 'volume']])
data.head()


# In[8]:


#split data into training and testing sets
from sklearn.model_selection import train_test_split
x = data[['open', 'low', 'high', 'volume']]
y = data['close']
x = x.values
y= y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#shape of training and testing sets
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(np.any(np.isnan(x)))
print(np.all(np.isfinite(x)))
print(type(x_train))


# In[9]:


#implement linear regression
#hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)
#error function
def error(X, y, theta):
    m = X.shape[0]
    y_ = hypothesis(X, theta)
    e = np.sum((y - y_)**2)
    return e/m
 
#gradient function
def gradient(X, y, theta):
    m = X.shape[0]
    y_ = hypothesis(X, theta)
    grad = np.dot(X.T, (y_ - y))
    return grad/m

#gradient descent function
def gradient_descent(X, y, learning_rate=0.1, max_iters=300):
    n = X.shape[1]
    theta = np.zeros((n,))
    error_list = []
    theta_list = []
    for i in range(max_iters):
        e = error(X, y, theta)
        error_list.append(e)
        grad = gradient(X, y, theta)
        theta = theta - learning_rate*grad
        theta_list.append(theta)
    return theta, error_list, theta_list


# In[10]:


ones = np.ones((x_train.shape[0], 1))
x_train = np.hstack((ones, x_train))
print(x_train.shape)
theta, error_list, theta_list = gradient_descent(x_train, y_train)


# In[11]:


print(theta_list)

#shape of theta_list
print(np.array(theta_list).shape)



# In[12]:


print(theta)


# In[13]:


#predict values
y_pred = hypothesis(x_train, theta)
#calculate r2 score
def r2_score(y, y_pred):
    num = np.sum((y - y_pred)**2)
    denom = np.sum((y - y.mean())**2)
    score = (1 - num/denom)
    return score*100
r2_score(y_train, y_pred)


# In[14]:


#predict values for test set
ones = np.ones((x_test.shape[0], 1))
x_test = np.hstack((ones, x_test))
print(x_test.shape)
y_pred = hypothesis(x_test, theta)
#calculate r2 score
r2_score(y_test, y_pred)


# In[ ]:




