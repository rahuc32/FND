#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('notes.csv')
v1= data["V1"]
v2 = data["V2"]

xys = [v1,v2]
xys = np.array(xys)
print(xys)
mean = np.mean(xys,1)
print(mean)
from sklearn.cluster import KMeans

std= np.std(xys,1)
import matplotlib.patches as patches
ellipse = patches.Ellipse([mean[0],mean[1]],std[0]*2,std[1]*2,alpha = 0.3)
fig,graph = plt.subplots()
graph.add_patch(ellipse)
graph.scatter(mean[0], mean[1])
graph.scatter(v1,v2,s= 0.0005, alpha= 0.01)
data = np.column_stack((v1,v2))

print(data)
km_res =KMeans(n_clusters = 1).fit(data) 
clusters = km_res.cluster_centers_
plt.scatter(v1,v2)
plt.scatter(clusters[:,0], clusters[:,1], s=100,alpha =0.6)
plt.xlabel("V1")
plt.ylabel("V2")


# In[4]:


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('notes.csv')
v1= data["V1"]
v2 = data["V2"]

data = np.column_stack((v1,v2))
print(data)
km_res =KMeans(n_clusters = 3).fit(v2) 
clusters = km_res.cluster_centers_
plt.scatter(v1,v2)
plt.scatter(clusters[:,0], clusters[:,1])

