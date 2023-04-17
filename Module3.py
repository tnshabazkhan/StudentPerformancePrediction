#!/usr/bin/env python
# coding: utf-8

# # Data Visualization 

# In[16]:


import numpy as np  
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


train_por = pd.read_csv('student-por.csv')
train_mat  = pd.read_csv('student-mat.csv')


# In[18]:


train_por.head()


# In[19]:


train_mat.head()


# In[20]:


train_por.shape


# In[21]:


train_por.dropna().shape #no null values in train_por


# In[22]:


train_mat.shape


# In[23]:


train_mat.dropna().shape #no null values in train_mat


# In[24]:


subset=train_por.columns


# In[25]:


train_por=train_por.drop_duplicates(subset=None, keep='first', inplace=False)
train_por.shape    #no duplicates in train_por


# In[26]:


subset=train_mat.columns


# In[27]:


train_mat=train_mat.drop_duplicates(subset=None, keep='first', inplace=False)
train_mat.shape    #no duplicates in train_mat


# In[28]:


train_por['subject']='Computerscience'


# In[29]:


train_mat['subject']='EC'


# In[30]:


train=pd.concat([train_por, train_mat], axis=0) #combining two data files with subject as discriminant column to distinguish columns


# In[31]:


train.head()


# In[32]:


train.to_csv('../students.csv', index=False)


# In[33]:


# contains all the merged data
data = pd.read_csv('../students.csv')


# In[34]:


def correlation(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(20, 15))
    colormap = sns.diverging_palette(150,50, as_cmap=True)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('Correlation.png', bbox_inches='tight')
    plt.show()


# In[35]:


correlation(data)


# In[36]:


data.columns


# In[37]:


data.to_csv('features.csv', index=False)


# In[38]:


data.shape

