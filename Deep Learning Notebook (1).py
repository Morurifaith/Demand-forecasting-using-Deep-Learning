#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

# Set seeds to make the experiment more reproducible.
#from tensorflow import set_random_seed
from numpy.random import seed


# In[2]:


#Loading data from csv
import csv
train=pd.read_csv(r'C:\Users\ALVIN\Desktop\train.csv')
test=pd.read_csv(r'C:\Users\ALVIN\Desktop\test.csv')
df=train.copy()
dt=test.copy()


# # EXPLORATORY DATA ANALYSIS

# In[3]:


train.head(5)


# In[4]:


train.shape


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


#maximum and minimum time period for train
print('Maximum time period:', train['date'].max())
print('Minimum time period:', train['date'].min())


# In[8]:


#maximum and minimum time period for test
print('Maximum time period:', test['date'].max())
print('Minimum time period:', test['date'].min())


# In[9]:


#grouping items sold daily and daily sales and store daily sales
daily_sales = train.groupby('date', as_index=False)['sales'].sum()
store_daily_sales = train.groupby(['store', 'date'], as_index=False)['sales'].sum()
item_daily_sales = train.groupby(['item', 'date'], as_index=False)['sales'].sum()


# In[10]:


store_daily_sales 


# In[11]:


#overall daily sales
daily_sales_sc = px.line(train,x=daily_sales['date'], y=daily_sales['sales'], title='Daily sales')
daily_sales_sc.show()


# In[12]:


#over store daily sales per store
store_daily_sales_sc=[]
for store in store_daily_sales['store'].unique():
    current_store_daily_sales=store_daily_sales[(store_daily_sales['store'] == store)]
    store_daily_sales_sc.append(go.Scatter(x=current_store_daily_sales['date'], y=current_store_daily_sales['sales'], name=('Store %s' % store)))
    
layout=go.Layout(title='Store daily sales',xaxis=dict(title='Date'),yaxis=dict(title='Sales'))
fig=go.Figure(data=store_daily_sales_sc,layout=layout)
iplot(fig)


# In[13]:


#over store daily sales per item
item_daily_sales_sc=[]
for item in item_daily_sales['item'].unique():
    current_item_daily_sales=item_daily_sales[(item_daily_sales['item'] == item)]
    item_daily_sales_sc.append(go.Scatter(x=current_item_daily_sales['date'], y=current_item_daily_sales['sales'], name=('Item%s' % item)))
    
layout=go.Layout(title='Item daily sales',xaxis=dict(title='Date'),yaxis=dict(title='Sales'))
fig=go.Figure(data=item_daily_sales_sc,layout=layout)
iplot(fig)


# In[14]:


def monthlyORyears_sales(data,time=['monthly','years']):
    data = train.copy()
    if time == "monthly":
        # Drop the day indicator from the date column:
        data.date = data.date.apply(lambda x: str(x)[:-3])
    else:
        data.date = data.date.apply(lambda x: str(x)[:4])
        
   # Sum sales per month: 
    data = data.groupby('date')['sales'].sum().reset_index()
    data.date = pd.to_datetime(data.date)
        
    return data


# In[15]:


m_df = monthlyORyears_sales(train,"years")
m_df.head(5)


# Transform the data into a time series problem

# In[30]:


def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[31]:


train = train[(train['date'] >= '2017-01-01')]
#Rearrange dataset so we can apply shift methods
train_gp = train.sort_values('date').groupby(['item', 'store', 'date'], as_index=False)
train_gp = train_gp.agg({'sales':['mean']})
train_gp.columns = ['item', 'store', 'date', 'sales']
train_gp.head()


# In[32]:


window = 29
lag = 1
series = series_to_supervised(train_gp.drop('date', axis=1), window=window, lag=lag)
series.head(20)


# In[33]:


last_item = 'item(t-%d)' % window
last_store = 'store(t-%d)' % window
series = series[(series['store(t)'] == series[last_store])]
series = series[(series['item(t)'] == series[last_item])]


# In[34]:


columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item', 'store']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item', 'store']]
series.drop(columns_to_drop, axis=1, inplace=True)
series.drop(['item(t)', 'store(t)'], axis=1, inplace=True)


# Train and Validation split

# In[35]:


labels_col = 'sales(t+%d)' % 1
labels = series[labels_col]
series = series.drop(labels_col, axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
X_train.head()


# In[36]:


epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)


# In[37]:


model_mlp = Sequential()
model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
model_mlp.add(Dense(1))
model_mlp.compile(loss='mse', optimizer=adam)
model_mlp.summary()


# In[38]:


mlp_history = model_mlp.fit(X_train.values, Y_train, validation_data=(X_valid.values, Y_valid), epochs=epochs, verbose=2)


# In[40]:


X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)


# In[41]:


model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer=adam)
model_lstm.summary()


# In[43]:


lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)


# In[44]:


mlp_train_pred = model_mlp.predict(X_train.values)
mlp_valid_pred = model_mlp.predict(X_valid.values)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, mlp_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, mlp_valid_pred)))


# In[46]:


lstm_train_pred = model_lstm.predict(X_train_series)
lstm_valid_pred = model_lstm.predict(X_valid_series)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, lstm_valid_pred)))


# In[ ]:




