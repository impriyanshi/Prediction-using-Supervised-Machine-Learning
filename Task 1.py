#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation
# # DATA SCIENCE AND BUSINESS ANALYTICS INTERN

# # BY: Priyanshi Jain
# 
# # TASK 1: Prediction Using Supervised ML
# 

# **In this task, we need to predict the percentage of a student based on the number of study hours. This task contains two variables i.e., 'format' is the number of hours studies and the 'percentage' is the percentage score of a student.This can be done using simple regression.**

# **Installing the modules**

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Data can be accessed and read from the given URL: http://bit.ly/w-data**

# In[16]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)


# **Extracting and Exploring the data**

# In[17]:


print(data.shape)
data.head()


# In[18]:


# checking for null values
data.isnull().sum()


# In[19]:


data.describe()


# In[20]:


data.info()


# In[21]:


data.plot(kind='scatter',x='Hours',y='Scores');
plt.show()


# In[22]:


data.corr(method='pearson')


# In[23]:


data.corr(method='spearman')


# In[24]:


hours=data['Hours']
scores=data['Scores']


# In[25]:


sns.distplot(scores)


# In[26]:


sns.distplot(hours)


# # Linear Regression

# In[27]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[28]:


import sklearn


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=50)


# In[30]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)


# In[31]:


m = reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[32]:


y_pred=reg.predict(X_test)


# In[33]:


import pandas as pd
actual_predicted=pd.DataFrame({'Percentage':y_test,'Predicted':y_pred})
actual_predicted


# In[34]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# **What would be the predicted score if a student studies for 8.5 hours/day?**

# In[35]:


h = 8.5
s=reg.predict([[h]])
print('If a student studies for {} hours/day, he/she wills score {} % in examination.'.format(h,s))


# # Model Evaluation

# In[36]:


import sklearn


# In[39]:


import scipy
from sklearn import metrics
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))

