#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from matplotlib.ticker import FixedLocator, FixedFormatter
from pylab import rcParams
import time


#Importing data
# data_raw = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\water-treatment.csv")
data_raw_st = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\Sales_Transactions_Dataset_Weekly.csv")
data_raw_st.head()


# In[14]:


data_raw_st=data_raw_st.drop(columns='Product_Code')
#Scaling the Data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(data_raw_st)
data_scale_st=scaler.transform(data_raw_st)
# data_scale=data_raw
# print(data_scale_st)
#Applying PCA to Data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_st=pca.fit_transform(data_scale_st)
# print(data_st)
plt.scatter(data_st[:,0], data_st[:,1])


# In[15]:


data_raw_wt = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\water-treatment.csv")
data_raw_wt.head()


# In[16]:


data_raw_wt=data_raw_wt.drop(columns='title')
data_raw_wt.head()

# Get ndArray of all column names 
data_raw_wt.columns.values
nrow,ncol=data_raw_wt.shape
for col_name in list(data_raw_wt.columns.values):
    data_raw_wt[col_name]=data_raw_wt[col_name].fillna(data_raw_wt[col_name].mean())


# In[17]:


scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(data_raw_wt)
data_scale_wt=scaler.transform(data_raw_wt)
#Applying PCA to Data
pca = PCA(n_components=2)
data_wt=pca.fit_transform(data_scale_wt)
# print(data_st)
plt.scatter(data_wt[:,0], data_wt[:,1])


# In[18]:


data_raw_live = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\live.csv")
data_raw_live.head()


# In[19]:


data_raw_live=data_raw_live.drop(columns='status_id')
data_raw_live=data_raw_live.drop(columns='status_type')
data_raw_live=data_raw_live.drop(columns='status_published')
data_raw_live=data_raw_live.drop(columns='Column1')
data_raw_live=data_raw_live.drop(columns='Column2')
data_raw_live=data_raw_live.drop(columns='Column3')
data_raw_live=data_raw_live.drop(columns='Column4')


# In[20]:


scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(data_raw_live)
data_scale_live=scaler.transform(data_raw_live)
#Applying PCA to Data
pca = PCA(n_components=2)
data_live=pca.fit_transform(data_scale_live)
plt.scatter(data_live[:,0], data_live[:,1])


# In[21]:


#Importing data
data_raw = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\Dow Jones Index.csv")

#Cleaning the Data
data_raw.describe(include='all')
data_raw.isnull().sum()
data_raw.percent_change_volume_over_last_wk = data_raw.percent_change_volume_over_last_wk.fillna(data_raw.percent_change_volume_over_last_wk.mean())
data_raw.previous_weeks_volume = data_raw.previous_weeks_volume.fillna(data_raw.previous_weeks_volume.mean())
data_raw.head()
# data_raw.columns[3:7]
data_raw[data_raw.columns[3:7]] = data_raw[data_raw.columns[3:7]].replace('[\$,]', '', regex=True).astype(float)
data_raw[data_raw.columns[11:13]] = data_raw[data_raw.columns[11:13]].replace('[\$,]', '', regex=True).astype(float)
data_raw.head()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data_raw['stock']= label_encoder.fit_transform(data_raw['stock'])
data_raw=data_raw.drop(columns='date')
data_raw.head()
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(data_raw)
data_scale=scaler.transform(data_raw)
# data_scale=data_raw
# print(data_scale)
#Applying PCA to Data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_dow=pca.fit_transform(data_scale)
# print(data_dow)
plt.scatter(data_dow[:,0], data_dow[:,1])


# In[22]:


from sklearn.cluster import KMeans
data_set_list=[data_dow,data_live,data_st,data_wt]
data_set_names=["Dow Jones dataset","Facebook Live dataset","Sales and Transactions Dataset","Water Treatment Dataset"]
optimum_k=[9,9,9,9]
optimum_eps=[0.71,0.71,0.71,0.71]
optimum_n=[9,9,9,9]
def calClusterCenters(data_set_sample,data_set_labels):
    clustcord=[]
    clus_list=[]
    for i in range(len(data_set_sample)):
        if(data_set_labels[i] not in clus_list):
            clus_list.append(data_set_labels[i])
    for i in clus_list:
        subclus=[]
        x_cord=sum(data_set_sample[data_set_labels==i,0])
        y_cord=sum(data_set_sample[data_set_labels==i,1])
        count=0
        for j in range(len(data_set_sample)):
            if(data_set_labels[j]==i):
                count+=1
        clustcord.append([x_cord/count,y_cord/count,i])
    return clustcord

def sumSquareError(data_set_sample,data_set_labels):
    sse=0
    data_set_centroids=calClusterCenters(data_set_sample,data_set_labels)
    for i in range(len(data_set_sample)):
        k=0
        for j in range(len(data_set_centroids)):
            if(data_set_centroids[j][2]==data_set_labels[i]):
                k=j
        x_dist=data_set_centroids[k][0]-data_set_sample[i,0]
        y_dist=data_set_centroids[k][1]-data_set_sample[i,1]
        total_dist=(x_dist*x_dist+y_dist*y_dist)
        sse+=total_dist
    return sse


# In[209]:


from matplotlib import cm
from tabulate import tabulate
tablelistkm=[]
fig, ay = plt.subplots(2,2,figsize=(10,10))
ax=[ay[0][0],ay[0][1],ay[1][0],ay[1][1]]
for i,data_set in enumerate(data_set_list):
    data_set_km= KMeans(n_clusters=optimum_k[i])
    data_set_stime=time.time()
    data_set_kmfit=data_set_km.fit(data_set)
    data_set_etime=time.time()
    data_set_kmfp=data_set_km.fit_predict(data_set)
    sse=data_set_kmfit.inertia_
    silhoute_score_dataset = silhouette_score(data_set, data_set_kmfit.labels_)
    data_set_ttime=data_set_etime-data_set_stime
    tablist=[data_set_names[i],silhoute_score_dataset,sumSquareError(data_set,data_set_kmfit.labels_),data_set_ttime]
    tablelistkm.append(tablist)
    
    cluster_labels = np.unique(data_set_kmfp)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(data_set, data_set_kmfp, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for x, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[data_set_kmfp == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(x) / n_clusters)
     ax[i].barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
             edgecolor='none', color=color)

     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    ax[i].axvline(silhouette_avg, color="red", linestyle="--")
    ax[i].set_title(data_set_names[i])
fig.suptitle("The CSM Plot for Different Dataset for KMeans", fontsize=14)  
plt.show()
print("The results for Kmeans are as follows:-")
print(tabulate(tablelistkm, headers=['Data Set','Silhouette Score','Sum Square Error','Time Taken']))    


# In[206]:


from sklearn.cluster import DBSCAN
tablelistdb=[]
fig, ay = plt.subplots(2,2,figsize=(10,10))
ax=[ay[0][0],ay[0][1],ay[1][0],ay[1][1]]
for i,data_set in enumerate(data_set_list):
    data_set_dbscan=DBSCAN(eps=optimum_eps[i],min_samples=4)
    db_stime=time.time()
    data_dbfit=data_set_dbscan.fit(data_set)
    db_etime=time.time()
    data_dbfp=data_set_dbscan.fit_predict(data_set)
    silhoute_score_dataset = silhouette_score(data_set, data_dbfit.labels_)
    data_db_ttime=db_etime-db_stime
#     db_sse=sumSqaureError(data_set,data_dbfit.labels_,data_dbfir.cluster_centers_)
    tablist=[data_set_names[i],silhoute_score_dataset,sumSquareError(data_set,data_dbfit.labels_),data_db_ttime]
    tablelistdb.append(tablist)
    
   
    cluster_labels = np.unique(data_dbfp)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(data_set, data_dbfp, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for x, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[data_dbfp == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(x) / n_clusters)
     ax[i].barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
             edgecolor='none', color=color)

     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    ax[i].axvline(silhouette_avg, color="red", linestyle="--")
    ax[i].set_title(data_set_names[i])
fig.suptitle("The CSM Plot for Different Dataset for DBSCAN Clustering", fontsize=14)  
plt.show()
print("The results for DBSCAN are as follows:-")
print(tabulate(tablelistdb, headers=['data set','Silhouette Score','Sum Square Error','Time Taken']))    


# In[207]:


from sklearn.cluster import AgglomerativeClustering
tablelistac=[]
fig, ay = plt.subplots(2,2,figsize=(10,10))
ax=[ay[0][0],ay[0][1],ay[1][0],ay[1][1]]
for i,data_set in enumerate(data_set_list):
    data_set_ac=AgglomerativeClustering(optimum_n[i])
    ac_stime=time.time()
    data_acfit=data_set_ac.fit(data_set)
    ac_etime=time.time()
    data_acfp=data_set_ac.fit_predict(data_set)
    silhoute_score_dataset = silhouette_score(data_set, data_acfit.labels_)
    data_ac_ttime=ac_etime-ac_stime
    tablist=[data_set_names[i],silhoute_score_dataset,sumSquareError(data_set,data_acfit.labels_),data_ac_ttime]
    tablelistac.append(tablist)
    
    cluster_labels = np.unique(data_acfp)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(data_set, data_acfp, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for x, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[data_acfp == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(x) / n_clusters)
     ax[i].barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
             edgecolor='none', color=color)

     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    ax[i].axvline(silhouette_avg, color="red", linestyle="--")
    ax[i].set_title(data_set_names[i])
fig.suptitle("The CSM Plot for Different Dataset for DBSCAN Clustering", fontsize=14)  
plt.show()
print("The results for Agglomerative Clustering are as follows:-")
print(tabulate(tablelistac, headers=['data set','Silhouette Score','Sum of Square Error','Time Taken']))  


# In[194]:


# # def sumSquareError():
# data_set_km= KMeans(n_clusters=9)
# data_dow_kmfit=data_set_km.fit(data_dow)
# print(data_dow_kmfit.labels_)
# print(calClusterCenters(data_dow,data_dow_kmfit.labels_))
# # print(data_dow_kmfit.labels_)
# print(data_set_kmfit.cluster_centers_)
# data_dow_kmeans=data_set_km.fit_predict(data_dow)
# # print(data_set_kmfit.cluster_centers_)
# # print(len(data_dow))
# # print(data_dow)

# data_color=['lightblue','green','blue','yellow','orange','violet','pink','grey','black']

# for i in range(9):  
#  plt.scatter(data_dow[data_dow_kmeans == i,0],
#             data_dow[data_dow_kmeans == i,1],
#             s=50, c=data_color[i],
#             marker='s', edgecolor='black',
#             label='cluster '+str(i+1))
# cust_center=data_dow_kmfit.cluster_centers_
# labels=data_set_kmfit.labels_
# # print(cust_center)
# # plt.scatter(data_dow_kmfit.cluster_centers_[:, 0],
# #             data_dow_kmfit.cluster_centers_[:, 1],
# #             s=250, marker='*',
# #             c='red', edgecolor='black',
# #             label='centroids')
# # for i in range(len(data_dow_kmfit.cluster_centers_)):
# plt.scatter(data_dow_kmfit.cluster_centers_[:, 0],
#             data_dow_kmfit.cluster_centers_[:, 1],
#             s=250, marker='*',
#             c='red', edgecolor='black',
#             label='centroids'+str(i+1))
# plt.legend(scatterpoints=1)
# plt.grid()
# plt.tight_layout()
# #plt.savefig('06_02.png', dpi=300)
# plt.show()
# print("The sum of square errors is:-",data_dow_kmfit.score(data_dow))


# In[ ]:





# In[ ]:





# In[ ]:




