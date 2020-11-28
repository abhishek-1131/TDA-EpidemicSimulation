import time
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.datasets import make_blobs
import numpy as np
# %matplotlib inline
import scipy
import time
import networkx as nx
import collections
# import EoN
from collections import defaultdict
import random
import pandas as pd
from seirsplus.models import *
from seirsplus.networks import *
import networkx

import cv2
import io
import PIL
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from PIL import Image

s = 10000
arr = [int(s/2), int(s/2)]
data,y = make_blobs(n_samples=arr,n_features=2,random_state=16, centers=[[1,1],[1,2]] ,cluster_std=0.5)

def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
SIGMA  = 1/5.2
GAMMA  = 1/10
MU_I   = 0.002

R0     = 2.5
BETA   = 1/(1/GAMMA) * R0
BETA_Q = 0.5*BETA

images_array = []
ppx = [10,20,30,40,50]
nn_list = [2,4,6,8,10]

for x in ppx:
    tsne = TSNE(n_components=2,perplexity=x)
    tsne_results = tsne.fit_transform(data)    
    for i in nn_list:
        A = kneighbors_graph(tsne_results, i, mode='connectivity', include_self=False, n_jobs=-1)
        Graph = nx.from_scipy_sparse_matrix(A)

        model_base = SEIRSNetworkModel(G = Graph, 
                                  beta = BETA, 
                                  sigma = SIGMA, 
                                  gamma = GAMMA, 
                                  initE = 1000, initI = 500)

        model_base.run(T=300)
        img = get_img_from_fig(model_base.figure_basic(ylim=0.2)[0])
        im = Image.fromarray(img)
        im.save(f'plots/pp={x}|nn={i}.png')