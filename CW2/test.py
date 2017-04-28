from   __future__             import print_function
from   sklearn.cluster        import KMeans
from   scipy.spatial.distance import cdist
from   sklearn.preprocessing  import normalize
from   scipy.spatial          import Voronoi, voronoi_plot_2d
from   scipy                  import stats
from   skimage                import io
import numpy                  as np
import matplotlib.pyplot      as plt
import itertools
import math
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from skimage.draw import polygon, circle
from sklearn.neighbors import KNeighborsClassifier




def read(char_name):
    f      = io.imread(char_name)   # read in image
    f_f    = np.array(f, dtype=float)
    z      = np.fft.fft2(f_f)           # do fourier transform
    q      = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq   =  np.log(np.absolute(q) +1)         # magnitude spectrum
    return Magq
    # return q




def sector1():
    n = np.zeros((400, 640), dtype=float)
    x = np.array([0, 100, 200])
    y = np.array([160, 0, 320])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    x = np.array([400, 300, 200])
    y = np.array([160, 0, 320])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    x = np.array([0, 100, 0])
    y = np.array([0, 0, 160])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    x = np.array([400, 300, 400])
    y = np.array([0, 0, 160])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    #plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n




def sector2():
    n = np.zeros((400, 640), dtype=float)
    x = np.array([0, 0, 200])
    y = np.array([160, 480, 320])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    x = np.array([400, 400, 200])
    y = np.array([160, 480, 320])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    #plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n

def bar():
    n = np.zeros((400, 640), dtype=float)
    x = np.array([170, 230, 230, 170])
    y = np.array([0, 00, 600, 600])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1
    #plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n

def ring():
    n = np.zeros((400, 640), dtype=float)
    xx, yy = circle(200, 320, 120)
    n[xx, yy] = 1
    xx, yy = circle(200, 320, 40)
    n[xx, yy] = 0
    plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    return n

def get_characters_features():
    mag = np.empty((400,640))
    features = np.empty((0, 4))
    labels = []
    for c in ['V','T','S']:
        for i in range(1,11):
            mag = read("chars/" + c + str(i) + ".GIF")
            feat = extract_features(mag)
            features = np.vstack((features, feat))
            labels.append(c)
    return features, labels

def get_myTest_characters_features():
    mag = np.empty((400,640))
    features = np.empty((0, 4))
    labels = []
    for c in ['V','T','S']:
        for i in range(1,5):
            mag = read("myTest/" + c + str(i) + ".GIF")
            feat = extract_features(mag)
            features = np.vstack((features, feat))
            labels.append(c)
    return features, labels


def extract_features(mag):
    space = np.zeros((4, 400, 640))
    sec1  = sector1()
    sec2  = sector2()
    bar1  = bar()
    ring1 = ring()
    space[0] = np.multiply(mag, sec1)
    space[1] = np.multiply(mag, sec2)
    space[2] = np.multiply(mag, bar1)
    space[3] = np.multiply(mag, ring1)
    sumed = ([np.sum(space[0]), np.sum(space[1]), np.sum(space[2]), np.sum(space[3])])
    return sumed




f, labels = get_characters_features()
targets = np.zeros(30)

t, labelsT = get_myTest_characters_features()
targetsT = np.zeros(12)

for i in range(len(labelsT)):
    if (labelsT[i] == 'V'):
        targetsT[i] = 0
    if (labelsT[i] == 'T'):
        targetsT[i] = 1
    if (labelsT[i] == 'S'):
        targetsT[i] = 2

for i in range(len(labels)):
    if (labels[i] == 'V'):
        targets[i] = 0 #blue
    if (labels[i] == 'T'):
        targets[i] = 1 #green
    if (labels[i] == 'S'):
        targets[i] = 2 #red

norm = f / 10000
print(norm)
print(targets)

normT = t / 10000
print(normT)
print(targetsT)

aChar = read("test/A1.GIF")
aFeat = extract_features(aChar)
bChar = read("test/B1.GIF")
bFeat = extract_features(bChar)
test = np.vstack((aFeat, bFeat))
aNorm = test / 10000
print(aNorm)

fe1 = 0
fe2 = 3

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(norm[:, [fe1,fe2]], targets)
h = 0.01  # step size in the mesh


x_min, x_max = norm[:, fe1].min() - 5, norm[:, fe1].max() + 5
y_min, y_max = norm[:, fe2].min() - 5, norm[:, fe2].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

#red - S

plt.scatter(norm[:, fe1], norm[:, fe2], c=targets, cmap=cmap_bold)
plt.scatter(normT[:, fe1], normT[:, fe2], c=targetsT, cmap=cmap_bold, marker="*", s=100)
#plt.scatter(aNorm[:, fe1], aNorm[:, fe2], c="y", cmap=cmap_bold, marker="*", s=500)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("feature " + str(fe1), fontsize=18)
plt.ylabel("feature " + str(fe2), fontsize=18)


plt.show()
