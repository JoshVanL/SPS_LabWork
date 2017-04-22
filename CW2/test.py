# for python 2 compatibility #
from __future__ import print_function
from scipy import signal
#                            #
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans

np.set_printoptions(threshold=np.inf)

def kimeans(Data, NClusters):
    km = KMeans(NClusters, n_init=1)
    fitted = km.fit(Data)
    print(fitted)
    return (fitted.cluster_centers_, fitted.labels_, fitted.inertia_)


i = 0
plotID =1;
# fig1 = plt.figure()
qs = np.zeros((400, 640))
Magqs = np.zeros((400, 640))


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
            if ((np.absolute(q[i][j])<10000)):
                q[i][j] = 0
                Magq[i][j] = 0

    #print(q[300])
    qs = np.concatenate((qs, q), axis=0)
    Magqs = np.concatenate((Magqs, Magq), axis=0)


    # ax1.imshow( np.log( np.absolute(q) + 1 ), cmap='gray' ) # io.
    plotID += 1

    w = np.fft.ifft2( np.fft.ifftshift(q) ) # do inverse fourier transform


    # fig2 = plt.figure()
    # ax2  = fig2.add_subplot( 111 )
    # ax2.axis('off')
    # ax2.imshow( np.array(w,dtype=int), cmap='gray' ) # io.
#
# for i in range(1,11):
#     f = io.imread('chars/T' + str(i) + '.GIF')   # read in image
#     f_f = np.array(f, dtype=float)
#     z = np.fft.fft2(f_f)           # do fourier transform
#     q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
#     Magq =  np.absolute(q)         # magnitude spectrum
#     Phaseq = np.angle(q)           # phase spectrum
#
#     #print (q)
#     # Usually for viewing purposes:
#     for i in range (q.shape[0]):
#         for j in range(q.shape[1]):
#             if ((np.absolute(q[i][j])<10000)):
#                 q[i][j] = 0
#                 Magq[i][j] = 0
#
#     #print(q[300])
#     qs = np.concatenate((qs, q), axis=0)
#     Magqs = np.concatenate((Magqs, Magq), axis=0)
#
# for i in range(1,11):
#     f = io.imread('chars/S' + str(i) + '.GIF')   # read in image
#     f_f = np.array(f, dtype=float)
#     z = np.fft.fft2(f_f)           # do fourier transform
#     q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
#     Magq =  np.absolute(q)         # magnitude spectrum
#     Phaseq = np.angle(q)           # phase spectrum
#
#     #print (q)
#     # Usually for viewing purposes:
    # for i in range (q.shape[0]):
    #     for j in range(q.shape[1]):
#             if ((np.absolute(q[i][j])<10000)):
#                 q[i][j] = 0
#                 Magq[i][j] = 0
#
#     #print(q[300])
#     qs = np.concatenate((qs, q), axis=0)
#     Magqs = np.concatenate((Magqs, Magq), axis=0)

def spectralRegion(data):
    feature1 = []
    feature2 = []
    center = (round(data.shape[0]/2), round(data.shape[1]/2))
    for i in range (round(data.shape[0]/2)):
        for j in range(data.shape[1]):
            if (not ((center[1] < y)):
                if (mag != 0):
                    mag = np.absolute(data[i][j])
                    if (math.tan(j/i)>= (math.pi/4) & (math.tan(j/i)<= (math.pi/2)):
                        feature1.append(mag)
                    if (math.tan(j/i)>= (math.pi/2) & math.tan(j/i)<= (3*math.pi/4)):
                        feature2.append(mag)







# f, Pxx_den = signal.periodogram(Magqs, signal.get_window('triang', 2.4))
# print (Pxx_den)
# plt.hist(Pxx_den)

# k = kimeans(qs, 2)
# print (k)
# nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(Magqs)
# distances, indices = nbrs.kneighbors(Magqs)
# print(indices)
# print(" ")
# print(distances)
print(np.absolute(q[0][0]))
print(q.shape)
plt.show()
