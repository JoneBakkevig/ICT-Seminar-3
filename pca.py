import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import sklearn.preprocessing as skp
from excelparse import fileToDframe
import os

filename = os.path.abspath('data.xlsx')

data = fileToDframe(filename, 0, )

# Normalize data for varying data sizes
min_max_scaler = skp.MinMaxScaler()
n_scaled = min_max_scaler.fit_transform(n)


def findNcomp(data):
    # run to find optimal number of components. 
    pca = sklearnPCA()
    pca.fit(data)
    # variance ratio as variable
    var = pca.explained_variance_ratio_
    # Create graph to determine best number of components
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(1,len(var)+1, 1.0), np.cumsum(var))
    plt.xticks(np.arange(0, len(var)+1, 1.0))
    plt.yticks(np.arange(0.1 ,1.1 , 0.1))
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio')
    plt.show()

def doPca(data, n_comp=None):

    pca = sklearnPCA(n_components=n_comp)
    pca.fit(data)

    if n_comp == None:
        findNcomp(data)

    return pca

target = clean_df.ix[:,6].values
print clean_df.ix[:,6].values

pca2 = doPca(n_scaled)
X_1 = pca2.transform(n_scaled)


pca3 = doPca(n_scaled, 3)
X_2 = pca3.transform(n_scaled)

ax1 = plt.scatter(X_1[:, 0], X_1[:, 1], c=target)
plt.colorbar(ax1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

ax2 = plt.scatter(X_2[:, 1], X_2[:, 2], c=target)
plt.colorbar(ax2)
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.show()

ax3 = plt.scatter(X_2[:, 0], X_2[:, 2], c=target)
plt.colorbar(ax3)
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.show()


cov = np.cov(n_scaled.T)

eigval, eigvec = np.linalg.eig(cov)

# Plotting weights
plt.hist(pca2.components_[0])
plt.hist(np.dot(np.array(n_scaled.T).transpose(),np.array(eigvec[:,1])))
plt.show()









