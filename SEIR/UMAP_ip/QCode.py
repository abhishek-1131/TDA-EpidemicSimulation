import networkx as nx
import matplotlib.pyplot as plt
import collections
import numpy as np
from sklearn.datasets import make_blobs
import umap.umap_ as umap
from seirsplus.models import *
from seirsplus.networks import *
import statistics
import seir
from seir_experiments import * 
import statistics
import pickle

def ipfnB(G1,s=10000,e=1,i=1):
#s = no of points
    SIGMA  = 1/5.2
    GAMMA  = 1/10
    MU_I   = 0.002
    
    R0     = 2.5
    BETA   = 1/(1/GAMMA) * R0
    BETA_Q = 0.5*BETA

    P      = 0.2
    Q      = 0.05
    
    E=s*e/100
    I=s*i/100
    
    model_base = SEIRSNetworkModel(G       = G1, 
                          beta    = BETA, 
                          sigma   = SIGMA, 
                          gamma   = GAMMA, 
                          initE = E, initI = I)
    
    model_base.run(T=300)
    fig,axs = model_base.figure_basic()
    
    line = axs.lines[1]
    p = line.get_data()
    p0 = p[0].compressed()
    p1 = p[1].compressed()
    #Maximum
    m = max(p[1])
    #Days Max
    ind= numpy.argmax(p[1])
    d = p[0][numpy.argmax(p[1])]
    #Mean and St Deviation
    mean  = statistics.mean(p1)
    stdev =  statistics.stdev(p1)
    line1 = axs.lines[2]
    w = line1.get_data()
    #Total Infections
    tot = 1 -(w[1][w[1].shape[0]-1])
    tp = 0.1*m
    np = 0.9*m
    #Ten Percent
    tpl =[]
    for i in range(len(p1)):
        x = numpy.abs(p1[i]-tp)
        if(x<0.001):
            #print(p1[i],x,p0[i])
            tpl.append(p0[i])
    tplval = statistics.mean(tpl)
    
    #Ninety Percent
    nplfh =[]
    nplsh = []
    for i in range(ind):
        x = numpy.abs(p1[i]-np)
        if(x<0.001):
            #print(p1[i],x,p0[i])
            nplfh.append(p0[i])
    for i in range(ind, len(p1)):
        x = numpy.abs(p1[i]-np)
        if(x<0.001):
            #print(p1[i],x,p0[i])
            nplsh.append(p0[i])
    npl=[(statistics.mean(nplfh)),(statistics.mean(nplsh))]
    
    
    return [m,d,tot,tplval,npl, mean, stdev],p

def calc_euclid(x,y):
    dist = sum((x-y)**2)

    return dist


def plot_degree_dist(min_degree, mean_degree, gammas):
    
    N=10000
    fig, ax = plt.subplots()
    deg_list=[]
    for gamma in gammas:
        degs = generate_power_law_degrees(N, min_degree, mean_degree, gamma)
        # Degree histogram
        ax = sns.distplot(degs, kde=False, label=gamma, ax=ax)
        # hist_kws={'alpha':0.2}, bins=np.arange(0, 1200, 20),
        deg_list.append(degs)
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of nodes')
    ax.legend(title='gamma')
    ax.set_title(f'Degree distribution (log scale) for N={N} nodes, mean={mean_degree}')
    print(np.median(degs), np.sort(degs)[-int(0.001 * N)])  # (median, top 0.1%)
    return ax,deg_list

#Sort Key
def mykey(x):
    return(x[0])

g1  = seir.generate_scale_free_graph(100000,min_degree=2,mean_degree=20,gamma=0.2)
g1 = nx.Graph(g1)

g2 = nx.barabasi_albert_graph(100000,m=10)

data1,y1 = make_blobs(n_samples=100000,n_features=2,cluster_std=3)
g3=nx.Graph()
g3.add_nodes_from(range(10000))
for i in range(len(data1)):

    dlist=[]
    for j in range(len(data1)):
        diff = calc_euclid(data1[i],data1[j])
        dlist.append([diff,j])
    dlist.sort(key=mykey)
    #print(dlist[:20])
    edgelist=[]
    cntr= 0
    p=1
    while(cntr!=deg[i] and p<10000):
    #for k in range(deg[i]):
        k= dlist[p][1]
        if(g3.degree[k]<deg[j]):
            edgelist.append((i,k))
            cntr=cntr+1
        p=p+1

    g3.add_edges_from(edgelist)
    print(i)
    
graphs=[g1,g2,g3]
with open('Final Outputs/Loc_BA_Ofir_Graphs_100k.pkl', 'wb') as outfile:
    pickle.dump(graphs, outfile, pickle.HIGHEST_PROTOCOL)


res1,line1 = ipfnB(g1,e=1,i=1)
res2,line2 = ipfnB(g2,e=1,i=1)
res3,line3 = ipfnB(g3,e=1,i=1)

results=[[res1,line1],[res2,line2],[res3,line3]]
with open('Final Outputs/Loc_v_Nloc_100k_lines.pkl', 'wb') as outfile:
    pickle.dump(results, outfile, pickle.HIGHEST_PROTOCOL)

