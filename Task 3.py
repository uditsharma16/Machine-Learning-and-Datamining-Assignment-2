

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
import tsne as tsn


#Importing data
data_raw = pd.read_csv(r"C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Assignment 2 Dataset\Sales_Transactions_Dataset_Weekly.csv")

print(data_raw.head())
print(data_raw.shape)
# pd.DataFrame(data_raw).to_numpy()
print(data_raw._get_numeric_data())
data_num=data_raw._get_numeric_data()
# print(data_raw)
# print(data_raw.dtype())
data_emb=TSNE(2).fit_transform(data_num)
print(data_emb)