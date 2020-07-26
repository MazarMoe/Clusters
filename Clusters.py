#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[3]:


Data = pd.read_csv(r"C:\Users\RayRay\Desktop\clusters.csv")


# In[4]:


Data


# In[6]:


plt.scatter(Data['Longitude'],Data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[9]:


x = Data.iloc[:,1:3]
kmeans = KMeans(3)
kmeans.fit(x)


# In[10]:


clusters = kmeans.fit_predict(x)
Data_with_clusters = Data.copy()
Data_with_clusters['Cluster'] = clusters
Data_with_clusters


# In[12]:


plt.scatter(Data_with_clusters['Longitude'],Data_with_clusters['Latitude'],c=Data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


# In[ ]:




