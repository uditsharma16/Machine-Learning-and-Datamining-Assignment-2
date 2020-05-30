
########################################Part 1###########################
#########################################1A##############################

import numpy as np
import pandas as pd
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
data_raw = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\Dow Jones Index.csv")

#printing data
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

# importing Kmeans clustering algorithm
from sklearn.cluster import KMeans
data_raw_kmeans = KMeans(n_clusters=3)
start_dow = time.time()
data_raw_kmeans.fit(data_raw)
end_dow = time.time()
timet_dow=end_dow-start_dow
data_raw.shape
data_raw_kmeans.cluster_centers_


print("Time take by Kmeans cluster",timet_dow)
#Calculating the sum square error or SSE -- the output is negative as python considers it as error but it is actually distance
sse=data_raw_kmeans.fit(data_raw).score(data_raw)
# print("The sum of squre errors:-",sse)
SSE = np.absolute(sse)
print("The sum of square errors is:-",SSE)
data_raw_clustered = pd.concat([data_raw,pd.Series(data_raw_kmeans.labels_)],axis=1)
data_raw_clustered.rename(columns={data_raw_clustered.columns[-1]:'Cluster_number'},inplace=True)
data_raw_clustered.describe(include='all')
data_raw_clustered.sort_values(['Cluster_number'])
silhoute_score = silhouette_score(data_raw, data_raw_kmeans.labels_)
print("The silhoute score is:",silhoute_score)

#K means per value of K
k_means_per_k = [KMeans(n_clusters=k,random_state=40).fit(data_raw) for k in range(1,10)]
# print(k_means_per_k)
silhoute_scores_per_k = [silhouette_score(data_raw,model.labels_) for model in k_means_per_k[1:]]
# print(silhoute_scores_per_k)
#ploting graph for silhoute_scores
rcParams['figure.figsize']=16,5
_ = plt.plot(range(2,10),silhoute_scores_per_k,'bo-',color='blue',linewidth=3,markersize=8,label='silhoute Curve')
_ = plt.xlabel('K',fontsize=14,family='Arial')
_ = plt.ylabel('Silhoute Score',fontsize=14,family='Arial')
_ = plt.grid(which='major',color='#cccccc',linestyle='--')
_ = plt.title('Silhoute curve for predicting optimal number of clusters', family='Arial',fontsize=14)

#optimal number of k
# k  = np.argmax(silhoute_scores_per_k)+3
# print(k)
k=2
opt_point=silhoute_scores_per_k[0]-silhoute_scores_per_k[1]
for i in range(7):
    diff_sil=silhoute_scores_per_k[i]-silhoute_scores_per_k[i+1]
    # print(diff_sil,i)
    if(diff_sil>opt_point):
        k=i
        opt_point=diff_sil

#line to mark optimal number of k in curve
_ = plt.axvline(x=k+2, linestyle='--', c='green',linewidth=3,label='Optimal Number of cluster ({})'.format(k))
_ = plt.scatter(k+2, silhoute_scores_per_k[k],c='red',s=400)
_ = plt.legend(shadow=True)
_ = plt.show()

################################################1-B########################################
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# data_raw = StandardScaler().fit_transform(data_raw)
# db=DBSCAN(eps=2, min_samples=3)
# data_dbscan=db.fit(data_raw)
# # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# # core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# print(labels)
#
# # from sklearn.cluster import KMeans
# # data_raw_kmeans = KMeans(n_clusters=11)
# #
# # data_raw_kmeans.fit(data_raw)
# # labels_true=data_raw_kmeans.labels_
# # print(labels_true)
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# # print("Adjusted Rand Index: %0.3f"
# #       % metrics.adjusted_rand_score(labels_true, labels))
# # print("Adjusted Mutual Information: %0.3f"
# #       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(data_raw, labels))

########################################1-C########################################
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(data_raw)
print('Cluster labels: %s' % labels)


# sse=ac.fit(data_raw).score(data_raw)
# # print("The sum of squre errors:-",sse)
# SSE = np.absolute(sse)
# print("The sum of square errors is:-",SSE)
data_raw_clustered = pd.concat([data_raw,pd.Series(ac.labels_)],axis=1)
data_raw_clustered.rename(columns={data_raw_clustered.columns[-1]:'Cluster_number'},inplace=True)
data_raw_clustered.describe(include='all')
data_raw_clustered.sort_values(['Cluster_number'])
silhoute_score = silhouette_score(data_raw, ac.labels_)
print("The silhoute score is:",silhoute_score)

#K means per value of K
ac_per_n = [AgglomerativeClustering(n_clusters=k).fit(data_raw) for k in range(1,10)]
silhoute_scores_per_n = [silhouette_score(data_raw,model.labels_) for model in ac_per_n[1:]]

#ploting graph for silhoute_scores
rcParams['figure.figsize']=16,5
_ = plt.plot(range(2,10),silhoute_scores_per_n,'bo-',color='blue',linewidth=3,markersize=8,label='silhoute Curve')
_ = plt.xlabel('K',fontsize=14,family='Arial')
_ = plt.ylabel('Silhoute Score',fontsize=14,family='Arial')
_ = plt.grid(which='major',color='#cccccc',linestyle='--')
_ = plt.title('Silhoute curve for predicting optimal number of clusters', family='Arial',fontsize=14)

#optimal number of k
k=2
opt_point=silhoute_scores_per_n[0]-silhoute_scores_per_n[1]
for i in range(7):
    diff_sil=silhoute_scores_per_n[i]-silhoute_scores_per_n[i+1]
    # print(diff_sil,i)
    if(diff_sil>opt_point):
        k=i
        opt_point=diff_sil

#line to mark optimal number of k in curve
_ = plt.axvline(x=k+2, linestyle='--', c='green',linewidth=3,label='Optimal Number of cluster ({})'.format(k))
_ = plt.scatter(k+2, silhoute_scores_per_n[k],c='red',s=400)
_ = plt.legend(shadow=True)
_ = plt.show()