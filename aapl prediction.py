#!/usr/bin/env python
# coding: utf-8

# In[48]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


aapl=pd.read_csv("C:/Users/jakey/OneDrive/Documents/personal projects/predicting AAPL stock price\AAPL.csv",index_col='Date')
aapl
#10409 days


# In[3]:


aapl.describe()


# In[4]:


#Check for any missing values values
aapl.isnull().values.any()


# In[5]:


aapl['Adj Close'].plot()


# In[6]:


#target variable=dependent variable
target=pd.DataFrame(aapl['Adj Close'])
#feature variables=independent variables
features=['Open','High','Low','Volume']
#Scaling data between 0 and 1 for precision and memory consumption
scaler=MinMaxScaler(feature_range=(0,1))
features_transform=scaler.fit_transform(aapl[features])
feature_transform=pd.DataFrame(columns=features,data=features_transform,index=aapl.index)
feature_transform


# In[51]:


#Split Training and Test sets with 80/20
test_ratio=.1
training_ratio=1-test_ratio
test_size=int(test_ratio*len(aapl))
training_size=int(training_ratio*len(aapl))
x_train,x_test=feature_transform[:training_size],feature_transform[training_size:]
y_train,y_test=target[:training_size],target[training_size:]
train_x,test_x=np.array(x_train),np.array(x_test)
#n x m -> n arrays containing 1 array with m elements
x_train=train_x.reshape(x_train.shape[0],1,x_train.shape[1])
x_test=test_x.reshape(x_test.shape[0],1,x_test.shape[1])


# In[52]:


lstm=Sequential()
lstm.add(LSTM(32,input_shape=(1,train_x.shape[1]),activation='relu',return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error',optimizer='adam')
history=lstm.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,shuffle=False)


# In[55]:


y_pred=lstm.predict(x_test)


# In[59]:


y_test['Predicted']=y_pred


# In[60]:


plt.figure(figsize=(50,10))
plt.plot(y_train['Adj Close'],label='Training Data')
plt.plot(y_test['Adj Close'],label='Test Data')
plt.plot(y_test['Predicted'],label='Prediction')
plt.legend()
plt.show()


# In[62]:


#more test data=more accuracy
print('The Mean Squared Error: ',mean_squared_error(y_test['Adj Close'].values,y_test['Predicted'].values))
print('The Mean Absolute Error: ',mean_absolute_error(y_test['Adj Close'].values,y_test['Predicted'].values))
print('The Root Mean Squared Error: ',np.sqrt(mean_squared_error(y_test['Adj Close'].values,y_test['Predicted'].values)))

