# for python 2 compatibility #
from __future__ import print_function
from scipy import signal
#                            #
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import math
# %matplotlib inline
from sklearn.neighbors import NearestNeighbors
#from sklearn import neighbors, datasets

from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

import itertools
#np.set_printoptions(threshold=np.inf)

def kimeans(Data, NClusters):
    km = KMeans(NClusters, n_init=50)
    fitted = km.fit(Data)
    print(fitted)
    return (fitted.cluster_centers_, fitted.labels_, fitted.inertia_)


i = 0
plotID =1;
# fig1 = plt.figure()
qs = np.zeros((400, 640))
qV = np.zeros((400, 640))
qS = np.zeros((400, 640))
qT = np.zeros((400, 640))
Magqs = np.zeros((400, 640))
PhaseqV = np.zeros((400, 640))
PhaseqT = np.zeros((400, 640))
PhaseqS = np.zeros((400, 640))
y=[]


for i in range(1,11):
    f = io.imread('chars/V' + str(i) + '.GIF')   # read in image
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum
    Phaseq = np.angle(q)           # phase spectrum

    # ax1  = fig1.add_subplot( 1, 11, plotID )
    # ax1.axis('off')
    #print (q)
    # Usually for viewing purposes:
    for i in range (q.shape[0]):
        for j in range(q.shape[1]):
            if ((np.absolute(q[i][j])<100)):
                q[i][j] = 0
                Magq[i][j] = 0

    #print(q[300])
    qs = np.concatenate((qs, q), axis=0)
    qV = np.concatenate((qV, q), axis=0)
    Magqs = np.concatenate((Magqs, Magq), axis=0)
    if (i == 1) :
        qV = q
    else :
        qV = np.concatenate((qV, q), axis=0)


    # ax1.imshow( np.log( np.absolute(q) + 1 ), cmap='gray' ) # io.
    plotID += 1

    w = np.fft.ifft2( np.fft.ifftshift(q) ) # do inverse fourier transform


    # fig2 = plt.figure()
    # ax2  = fig2.add_subplot( 111 )
    # ax2.axis('off')
    # ax2.imshow( np.array(w,dtype=int), cmap='gray' ) # io.

for i in range(1,11):
    f = io.imread('chars/T' + str(i) + '.GIF')   # read in image
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum
    Phaseq = np.angle(q)           # phase spectrum

    #print (q)
    # Usually for viewing purposes:
    for i in range (q.shape[0]):
        for j in range(q.shape[1]):
            if ((np.absolute(q[i][j])<10000)):
                q[i][j] = 0
                Magq[i][j] = 0

    #print(q[300])
    qs = np.concatenate((qs, q), axis=0)
    qT = np.concatenate((qT, q), axis=0)
    Magqs = np.concatenate((Magqs, Magq), axis=0)
    if (i == 1) :
        qT = q
    else :
        qT = np.concatenate((qT, q), axis=0)

for i in range(1,11):
    f = io.imread('chars/S' + str(i) + '.GIF')   # read in image
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum
    Phaseq = np.angle(q)           # phase spectrum

    #print (q)
    # Usually for viewing purposes:
    for i in range (q.shape[0]):
        for j in range(q.shape[1]):
            if ((np.absolute(q[i][j])<10000)):
                q[i][j] = 0
                Magq[i][j] = 0

    #print(q[300])
    qs = np.concatenate((qs, q), axis=0)
    qS = np.concatenate((qS, q), axis=0)
    Magqs = np.concatenate((Magqs, Magq), axis=0)
    if (i == 1) :
        qS = q
    else :
        qS = np.concatenate((qS, q), axis=0)

def spectralRegion(data, index):
    feature1 = []
    feature2 = []
    label = []
    center = (round(data.shape[0]/2), round(data.shape[1]/2))
    for j in range (round(data.shape[1]/2)):
        for i in range(data.shape[0]):
            mag = np.absolute(data[i][j])
            if (mag != 0):
                n = center[0] - i
                m = center[1] - j
                if (j != 0):
                    if (math.tan(i/j) >= (math.pi/8) and  math.tan(i/j) <= (math.pi/4 + math.pi/8)):
                        feature1.append(mag)
                if (i != 0):
                    if (math.tan(j/i) >= (math.pi/8) and math.tan(j/i) <= (math.pi/4 + math.pi/8)):
                        feature2.append(mag)
                        y.append(index)
    #print(feature1)
    #print(feature2)
    f = list(map(lambda x, y: [x,y], feature1, feature2))
    features = np.matrix(f)
    print(features)
    return features




# f, Pxx_den = signal.periodogram(Magqs, signal.get_window('triang', 2.4))
# print (Pxx_den)
# plt.hist(Pxx_den)

#k = kimeans(qs, 3)
# print (k)
# nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(Magqs)
# distances, indices = nbrs.kneighbors(Magqs)
# print(indices)
# print(" ")
# print(distances)
#print(np.absolute(q[0][0]))
#print(q.shape)

#qS = np.concatenate((qS, q), axis=0)
y = []
featureV = spectralRegion(qV, 0)
featureT = spectralRegion(qT, 1)
featureS = spectralRegion(qS, 2)
#for i in range(0, featureV.shape[0]):
#    y.append(0)
#for i in range(featureV.shape[0], featureT.shape[0]):
#    y.append(1)
#for i in range(featureV.shape[0] + featureT.shape[0], featureS.shape[0]):
#    y.append(2)
features = featureV
features = np.concatenate((features, featureT), axis=0)
features = np.concatenate((features, featureS), axis=0)

#k = kimeans(features, 3)

n_neighbors = 15

# import some data to play with
#X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
#y = k[1]
#
h = 1000  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we create an instance of Neighbours Classifier and fit the data.
#clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', n_jobs=-1)
#clf.fit(features, y)
clf = NearestNeighbors(n_neighbors=15)
clf.fit(features) 
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = features[:, 0].min() - 1, (features[:, 0].max() + 1)
y_min, y_max = features[:, 1].min() - 1, (features[:, 1].max() + 1)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = neigh.kneighbors_graph(features)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
#plt.scatter(features[:, 0], features[:, 1], c=y, cmap=cmap_bold)
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
#plt.title("3-Class classification (k = %i, weights = '%s')"
#          % (n_neighbors, 'distance'))

plt.show()

#clusters = k[1]
#
#indeces   = [[] for x in range(3)]
#cluster   = [[] for x in range(3)]
#
#clrs     = itertools.cycle(["r", "g", "b", "c", "m"])
#
#for i in range (3):
#    indeces[i] = np.where(clusters==i)
#    cluster[i] = list(map(lambda x: features[x], indeces[i]))
#print(k)
#
#for i in range(3):
#    plt.scatter(cluster[i][0].take((0,), axis=1), cluster[i][0].take((1,), axis=1), color=next(clrs))
#plt.show()

