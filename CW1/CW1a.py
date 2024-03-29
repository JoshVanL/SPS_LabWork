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

def kimeans(Data, NClusters):
    km = KMeans(NClusters, n_init=1)
    fitted = km.fit(Data)
    print(fitted)
    return (fitted.cluster_centers_, fitted.labels_, fitted.inertia_)

def initList(num):
    return [[] for x in range(num)]


data = np.loadtxt('trainingSet.dat')
print(data)
print(data.shape)

#plotmatrix(data)

NClusters = 3

X = initList(NClusters)
X = np.full((150, 2), 0.)
for i in range(0,150):
    X[i] = np.array([data[i][0], data[i][3]])
print(X)
print(X.shape)

max = 0
for i in range(600):
    testing = kimeans(X, 3)
    if(testing[2] >= max):
        max = testing[2]
        classLabels = testing


print(classLabels)
fittedLabels = (classLabels[1])
print((fittedLabels.shape))
fittedCenters  = np.array(classLabels[0])
print(fittedCenters)

print(fittedLabels)
indices = np.array(np.where(fittedLabels ==0))
indices = indices.reshape(indices.shape[1])
X1 = np.arange(100.).reshape(50,2)
print(indices.shape)
for i in range(indices.shape[0]):
    X1[i] = X[indices[i]]
print(X1)


indices = np.array(np.where(fittedLabels ==1))
indices = indices.reshape(indices.shape[1])
X2 = np.arange(100.).reshape(50,2)
print(indices.shape)
for i in range(indices.shape[0]+1):
    X2[i] = X[indices[i]]


indices = np.array(np.where(fittedLabels ==2))
indices = indices.reshape(indices.shape[1])
X3 = np.arange(100.).reshape(50,2)
print(indices.shape)
for i in range(indices.shape[0]+1):
    X3[i] = X[indices[i]]
print("here")
print(X1)
print(X2)
print(X3)


fig = plt.figure()
bx = fig.add_subplot( 111 )
bx.scatter(X1[:,0], X1[:,1], c='b')
bx.scatter(X2[:,0], X2[:,1],c='r')
bx.scatter(X3[:,0], X3[:,1],c='g')
#plt.show()


testData = np.loadtxt('testSet.dat')
print(testData.shape)
T = np.full((15, 2), 0.)

for i in range(0,15):
    T[i] = np.array([data[i][0], data[i][3]])

vec = cdist(T,fittedCenters, 'euclidean', p=2) 

print(vec)

P1 = np.empty((0,2))
P2 = np.empty((0,2))
P3 = np.empty((0,2))

print(T.shape)
print(T[0].shape)
print(P1.shape)

for i in range(0,15):
    v = np.argmin(vec[i])
    if (v == 0):
        P1 = np.vstack((P1, T[i]))
    elif (v == 1):
        P2 = np.vstack((P2, T[i]))
    else :
        P3 = np.vstack((P3, T[i]))

print(P1)
print(P2)
print(P3)

#fig2 = plt.figure()
bx = fig.add_subplot( 111 )
bx.scatter(P1[:,0], P1[:,1], c='b', marker="*")
bx.scatter(P2[:,0], P2[:,1], c='r', marker="*")
bx.scatter(P3[:,0], P3[:,1], c='g', marker="*")
vor1 = Voronoi(fittedCenters)
voronoi_plot_2d(vor1, bx)
plt.show()
clusters = np.array([X1, X2, X3])
