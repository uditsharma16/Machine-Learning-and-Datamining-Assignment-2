
########################################Part 1###########################
# from sklearn.datasets import load_iris
# X, y = load_iris(return_rawdata_y=True)
#
# # kmeans = KMeans(n_clusters=2).fit(rawdata)
# # kmeans.labels_

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
#importing Kmeans clustering algorithm

from sklearn.cluster import KMeans

data_raw_kmeans = KMeans(n_clusters=3)
data_raw_kmeans.fit(data_raw)
data_raw.shape
data_raw_kmeans.cluster_centers_

