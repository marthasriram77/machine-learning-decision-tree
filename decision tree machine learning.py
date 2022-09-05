#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


# In[26]:


#import dataset
df=pd.read_csv(r"Downloads\news.csv")


# In[27]:


df


# In[28]:


df.info()


# In[29]:


df.isnull().sum()


# In[30]:


print("There are {} rows and {} columns.".format(df.shape[0],df.shape[1]))


# In[31]:


df.describe()


# In[32]:


#Let's drop unimportant columns
df=df.drop(['Unnamed: 0'],axis=1)


# In[33]:


df


# In[34]:


df['label'].value_counts()


# In[35]:


plt.figure(figsize=(5,10));
sns.countplot(df['label']);


# In[36]:


x = df.iloc[ : , :-1].values
y = df.iloc[ : , -1].values


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(stop_words="english",max_features=1000)


# In[38]:


x1=vect.fit_transform(x[:,0]).todense()
x2=vect.fit_transform(x[:,1]).todense()
x_mat=np.hstack((x1,x2))


# In[39]:


x_mat


# In[46]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x_mat,y,random_state=1000)


# In[48]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(x_train,y_train)


# In[53]:


y_pred=model.predict(x_test)


# In[55]:


y_pred


# In[56]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy=accuracy_score(y_pred,y_test)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[ ]:




