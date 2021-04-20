
# coding: utf-8

# # GRIP: THE SPARK FOUNDATION
# 
# 

# ## Data Science and Business Analytics Intern

# ## Author Uvesh Khan
# 

# ## TasK 1: Prediction using Supervised ML

# Predict the percentage of an student based on the no. of study hours.

# In[41]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data= pd.read_csv(url)


# Exploring Data

# In[43]:


print(data.shape)
data.head()


# In[44]:


data.describe()


# In[45]:


data.info()


# In[46]:


data.plot(kind='scatter',x='Hours',y='Scores')
plt.show()


# In[47]:


data.corr(method='pearson')


# In[48]:


data.corr(method='spearman')


# In[49]:


hours= data["Hours"]
scores=data["Scores"]


# In[18]:


sns.displot(hours)


# In[19]:


sns.displot(scores)


# Linear Regression

# In[50]:


X= data.iloc[:,:-1].values
y= data.iloc[:,-1].values


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=50,)


# In[52]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)


# In[54]:


m=reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# In[55]:


y_pred=reg.predict(X_test)


# In[58]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[60]:


sns.set_style('whitegrid')
sns.histplot(np.array(y_test-y_pred))
plt.show()


# What would be the prediction score if student studies for 9.25 hours/days?

# In[61]:


h=9.25
s=reg.predict([[h]])
print('If a student studies for {} hours per day he will score {} % in exam.'.format(h,s))


# Model Evaluation

# In[63]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred))
print('R2 Score: ',r2_score(y_test,y_pred))

