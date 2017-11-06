import os
import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import IncrementalPCA as IPCA
import sklearn.ensemble as ens
from sklearn.model_selection import cross_val_score
import sklearn.preprocessing as skp
import sklearn.metrics as skm
import sklearn.svm as svm


mndata = MNIST(os.path.abspath('../python-mnist/data'))

# Preprocessing
train, labels = mndata.load_training()
test, labels2 = mndata.load_testing()


# Normalizing
min_max_scaler = skp.MinMaxScaler()
n_train = min_max_scaler.fit_transform(train)
n_test = min_max_scaler.fit_transform(test)

def find_n_comp(data):

    pca = sklearnPCA()
    pca.fit(data)

    var = pca.explained_variance_ratio_[:100]
    # Plot function for explained variance per component
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(1, len(var) + 1, 1.0), np.cumsum(var))
    plt.xticks(np.arange(0, len(var) + 1, 1.0))
    plt.yticks(np.arange(0.1, 1.1, 0.1))
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio')
    plt.show()

def do_pca(train_d, test_d, n_comp=None):

    pca = sklearnPCA(n_components=n_comp, whiten=True, svd_solver='full')
    train_x = pca.fit_transform(train_d)
    test_x = pca.transform(test_d)

    return pca, train_x, test_x



X = data = np.array(n_train)
Y = target = np.array(n_test)
images = np.reshape(data, (data.shape[0],28,28))


def plot_embedding(X, title=None):
    # Plot function for visualization of classification
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})


    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.show()

def add_noise(data, sigma=32.0, size=None):
    # Noise function, sigma 16.0 ~ 6% noise
    if not size:
        size = data.shape
    noise = np.random.normal(0.0, sigma, size)
    return np.clip(data+noise,0,255)


# PCA and plotting for noisy dataset
#pca2, x3, x4 = do_pca(add_noise(data),target)
# Plot visualization
#plot_embedding(x3)

# PCA and plotting for original dataset
pca1, x1, x2 = do_pca(n_train,n_test,0.8)
# Plot visualization
#plot_embedding(x1)

# SVM classification for scoring original estimator (PCA)
svm1 = svm.NuSVC(verbose=True)
svm1.fit(x2[2000:], list(labels2)[2000:])
svm_score = svm1.score(x2[:2000], list(labels2)[:2000])

print svm_score

