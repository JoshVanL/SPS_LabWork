import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import stats
import matplotlib.pyplot as plt


import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 24

def plotmatrix(Matrix):
    r, c = Matrix.shape
    fig = plt.figure()
    plotID = 1
    for i in range(c):
        for j in range(c):
            ax = fig.add_subplot( c, c, plotID )
            ax.scatter( Matrix[:,i], Matrix[:,j] )
            plotID += 1
    plt.show()

data = np.loadtxt('testSet.dat')
print(data)
print(data.shape)

X = np.array([data[:,2], data[:,3]])
X = np.reshape(X, (2,15))
print(X)
print(X.shape)

def kmeans(Data, NClusters):
    km = KMeans(NClusters)
    fitted = km.fit(Data)
    return (fitted.cluster_centers_, fitted.labels_, fitted.inertia_)

classLabels = kmeans(X, 2)

print(classLabels)
