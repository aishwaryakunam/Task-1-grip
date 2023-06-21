#!/usr/bin/env python
# coding: utf-8

# # Aishwarya Kunam

# In[5]:


#importing all libraries required in this notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#plotting data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df.head(10)


# In[9]:


df.info()


# In[10]:


df.shape


# In[12]:


df.describe()


# In[19]:


#plotting distribution of score vs hours
df.plot.bar(x="Hours", y="Scores", style="o")
plt.title("Hours vs Score")
plt.xlabel("Study hrs of the students")
plt.ylabel("score obtained")
plt.show()


# In[29]:


x = df.iloc[:,:-1].values
y =  df.iloc[:,1].values


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)


# In[37]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("Training complete.")


# In[40]:


#plotting the regression line
line = regressor.coef_*x+regressor.intercept_

#plotting for the test data
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[41]:


print(x_test) #Testing data - In Hrs
y_pred = regressor.predict(x_test) #Predicting the scores


# In[43]:


prediction = pd.DataFrame({"Actual" : y_test, "predicted" : y_pred})
prediction


# In[51]:


own_pred = regressor.predict([[9.25]])
print("The predicted score if a studnet studies for a 9.25hr/day is", own_pred[0])


# In[55]:


from sklearn import metrics
from sklearn.metrics import r2_score
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2_score = metrics.r2_score(y_test, y_pred)

print("Mean Absolute Error:", MAE)
print("Mean Squrae Error:", MSE)
print("Root Mean Squared Error:", RMSE)
print("R2 score:",R2_score)

