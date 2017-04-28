
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
from skimage.draw import polygon
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

    plt.matshow(n, fignum=100, cmap=plt.cm.gray)
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

    plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n




def get_characters_features():
    mag = np.empty((400,640))
    features = np.empty((0, 2))
    labels = []
    for c in ['V','T','S']:
        for i in range(1,11):
            mag = read("chars/" + c + str(i) + ".GIF")
            feat = extract_features(mag)
            features = np.vstack((features, feat))
            labels.append(c)
    return features, labels



def extract_features(mag):
    space = np.zeros((2, 400, 640))
    sec1 = sector1()
    sec2 = sector2()
    space[0] = np.multiply(mag, sec1)
    space[1] = np.multiply(mag, sec2)
    sumed = ([np.sum(space[0]), np.sum(space[1])])
    return sumed




f, labels = get_characters_features()
targets = np.zeros(30)





for i in range(len(labels)):
    if (labels[i] == 'V'):
        targets[i] = 0
    if (labels[i] == 'T'):
        targets[i] = 1
    if (labels[i] == 'S'):
        targets[i] = 2

norm = f / f.max(axis=0)
print(norm)
print(targets)





clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(norm, targets)
h = 0.001  # step size in the mesh

x_min, x_max = norm[:, 0].min() - .1, norm[:, 0].max() + .1
y_min, y_max = norm[:, 1].min() - .1, norm[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(norm[:, 0], norm[:, 1], c=targets, cmap=cmap_bold)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("sector 1", fontsize=18)
plt.ylabel("sector 2", fontsize=18)


aChar = read("test/A1.GIF")
aFeat = extract_features(aChar)
bChar = read("test/B1.GIF")
bFeat = extract_features(bChar)
test = np.vstack((aFeat, bFeat))
aNorm = test / test.max(axis=0)
print(aNorm)
plt.scatter(aNorm[:, 0], aNorm[:, 1], c="teal", cmap=cmap_bold, marker="*")


plt.show()

