import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import stats
import matplotlib.pyplot as plt


#%matplotlib inline
# notebook
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 24

data = np.loadtxt('trainingSet.dat')
print(data)
print(data.shape)

mu = np.mean(data, axis=1)
cov = np.cov(data)
print(mu)
#print(cov)
a = 
arr = np.random.multivariate_normal(mu, cov)
print(arr)

