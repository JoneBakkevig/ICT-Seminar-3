import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import sklearn.preprocessing as skp
from excelparse import fileToDframe
import os
from random import random
from mnist import MNIST
mndata = MNIST(os.path.abspath('../python-mnist/data'))
images, labels = mndata.load_training()


columns = [3, 4, 5, 7, 15, 20, 35, 36, 43, 66, 79]
#columns_ = [3, 23, 25]

filename = os.path.abspath('../Data_nieuwveer.xlsx')

df1 = fileToDframe(filename, 92, columns, 1)
#df2 = fileToDframe(filename, 2, columns_, 3)


# Concatenating the frames from the separately parsed sheets
#df = pd.concat([df1, df2], axis=1)

# Dropping NaN values
clean_df = df1.dropna(subset=['COD','COD.1','COD.2','Precipitation'])
data = clean_df.reset_index(drop=True)
mu, sigma = 0, 0.1
for i in range(25):
    noise = np.random.normal(mu, sigma, clean_df.shape)

    df = clean_df+noise
    df = df.abs()

    data = data.append(df)


vol = 3500.0


data['Recirc_factor'] = (data['Qeffl_recirc']/(data['Qinf_total']-data['Qeffl_recirc']))
data['HRT'] = vol/(data['Qinf_total']+data['filtrate'])

print data.head()

data.to_excel('data.xlsx',startrow=0,startcol=0)







