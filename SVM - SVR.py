#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # Importing the dataset

# In[9]:


df = pd.read_csv('SampleData.csv')


# # EDA

# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.describe()


# In[13]:


df.info()


# In[16]:


df.columns


# In[17]:


# df.columns = ['Hours','Marks']
df.rename(columns={'Hours of Study':'Hours'}, inplace=True)


# In[18]:


df.isnull().sum()


# In[19]:


plt.scatter(df.Hours, df.Marks)
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.title('Hours of Study V/s Marks')


# In[22]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, cmap='copper')


# In[23]:


plt.plot(df.Hours,df.Marks)


# # Feature Scalling
# 
# * Standardization
# * Normalisation

# In[24]:


df.head()


# In[ ]:


-1 to 1


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[37]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[38]:


stanscale = StandardScaler()


# In[40]:


X = stanscale.fit_transform(X.reshape(-1,1))
y = stanscale.fit_transform(y.reshape(-1,1))


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                    random_state=10)


# In[43]:


X_train.shape


# In[44]:


from sklearn.svm import SVR


# In[47]:


model = SVR(kernel='rbf')


# In[48]:


model.fit(X_train,y_train)


# In[49]:


y_pred = model.predict(X_test)


# In[50]:


y_pred


# In[53]:


y_pred = stanscale.inverse_transform(y_pred)


# In[54]:


y_pred


# In[55]:


y_test = stanscale.inverse_transform(y_test)


# In[56]:


y_test


# In[57]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual Marks V/s Predicted Marks')


# In[64]:


stanscale.inverse_transform(model.predict([[5]]))


# In[65]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[66]:


model.score(X_test,y_test)


# In[67]:


r2_score(y_test, y_pred)


# In[68]:


mean_squared_error(y_test, y_pred)


# In[69]:


mean_absolute_error(y_test, y_pred)

