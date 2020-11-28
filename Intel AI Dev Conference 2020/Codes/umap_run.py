import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
sns.set(style = 'white',context= 'notebook', rc = {'figure.figsize':(14,10)})

start = time.time()

import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
reducer = umap.UMAP()

s = 1250000
arr = [int(0.2*s),int(0.05*s),int(0.1*s),int(0.16*s),int(0.12*s),int(0.1*s),int(0.04*s),int(0.08*s),int(0.15*s)]
data,y = make_blobs(n_samples=arr,n_features=2,random_state=16, cluster_std=1)
time1= time.time()
print("Data creation: ",time1-start)

embedding = reducer.fit_transform(data)
time2 = time.time()
print("Create Embedding:",time2-time1,"   ||| From start: ",time2-start)

plt.scatter(embedding[:,0],embedding[:,1], c=y)
time3 = time.time()
print("Plot:",time3-time2)
print("Total: ",time3-start)