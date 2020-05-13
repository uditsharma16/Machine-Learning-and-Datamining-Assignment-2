import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
path=r'C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\Dow Jones Index.csv'
rawdata = pd.read_csv(path)
print(rawdata)
########################################Part 1###########################
# from sklearn.datasets import load_iris
# X, y = load_iris(return_rawdata_y=True)
#
# # kmeans = KMeans(n_clusters=2).fit(rawdata)
# # kmeans.labels_