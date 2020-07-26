#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[5]:


Data = pd.read_csv(r"C:\Users\RayRay\Desktop\data.csv", index_col = 0)


# In[6]:


Data


# In[7]:


Data.describe()


# In[8]:


Data.corr()


# In[14]:


plt.figure(figsize = (12, 9))
s = sns.heatmap(Data.corr(),
               annot = True, 
               cmap = 'RdBu',
               vmin = -1, 
               vmax = 1)
plt.title('Correlation Heatmap')
plt.show()

