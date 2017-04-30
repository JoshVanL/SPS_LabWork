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
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal


###################################################################
##
## Read image data
##
###################################################################

def read(char_name):
    f      = io.imread(char_name)   # read in image
    f_f    = np.array(f, dtype=float)
    z      = np.fft.fft2(f_f)           # do fourier transform
    q      = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq   =  np.log(np.absolute(q) +1)         # magnitude spectrum

    ## Uncomment to see fourier space
    ##fig1 = plt.figure()
    ##ax1  = fig1.add_subplot( 111 )
    ##ax1.axis('off')
    ### Usually for viewing purposes:
    ##ax1.imshow( np.log( np.absolute(q) + 1 ), cmap='gray' ) # io.

    return Magq
    # return q



###################################################################
##
## Sector 1 polygon
## Diaganal tirangles
##
###################################################################

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

    ## Uncomment to view feature polygon
    #plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n



###################################################################
##
## Sector 2 polygon
## Vertical triangles
##
###################################################################

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

    ## Uncomment to view feature polygon
    #plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n

###################################################################
##
## Bar polygon
##
###################################################################

def bar():
    n = np.zeros((400, 640), dtype=float)
    x = np.array([170, 230, 230, 170])
    y = np.array([0, 0, 640, 640])
    xx, yy = polygon(x, y)
    n[xx, yy] = 1

    # Uncomment to view feature polygon
    # plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    # plt.show()
    return n

###################################################################
##
## Ring polygon
##
###################################################################

def ring():
    n = np.zeros((400, 640), dtype=float)
    xx, yy = circle(200, 320, 120)
    n[xx, yy] = 1
    xx, yy = circle(200, 320, 40)
    n[xx, yy] = 0

    ## Uncomment to view feature polygon
    #plt.matshow(n, fignum=100, cmap=plt.cm.gray)
    #plt.show()
    return n

###################################################################
##
## Extract training data features
##
###################################################################

def get_char_features():
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



###################################################################
##
## Extraxt test data characters
##
###################################################################

def get_myTest_char_features():
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




###################################################################
##
## Extract features (sector 1, sector 2, bar and ring)
##
###################################################################

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



###################################################################
##
## Extract training data
##
###################################################################


f, labels = get_char_features()
targets = np.zeros(30)

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


###################################################################
##
## Extract test data
##
###################################################################

t, labelsT = get_myTest_char_features()
targetsT = np.zeros(12)

for i in range(len(labelsT)):
    if (labelsT[i] == 'V'):
        targetsT[i] = 0
    if (labelsT[i] == 'T'):
        targetsT[i] = 1
    if (labelsT[i] == 'S'):
        targetsT[i] = 2


normT = t / 10000
print(normT)
print(targetsT)


###################################################################
##
## Extract A and B character features
##
###################################################################

aChar = read("test/A1.GIF")
aFeat = extract_features(aChar)
bChar = read("test/B1.GIF")
bFeat = extract_features(bChar)
test = np.vstack((aFeat, bFeat))
#aNorm = test / 10000
print(aFeat)
aNorm = np.empty(4)
for i in range(4):
    aNorm[i] = aFeat[i] / 10000
bNorm = np.empty(4)
for i in range(4):
    bNorm[i] = bFeat[i] / 10000
#aNorm = aFeat / 10000
print(aNorm)



###################################################################
##
## feature selection
## 0 - sector 1
## 1 - sector 2
## 2 - bar
## 3 - ring
##
###################################################################
fe1 = 0
fe2 = 3



###################################################################
##
## Applying nearest neighbour classifier
##
###################################################################


clf = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
clf.fit(norm[:, [fe1,fe2]], targets)
h = 0.01  # step size in the mesh


x_min, x_max = norm[:, fe1].min() - 5, norm[:, fe1].max() + 5
y_min, y_max = norm[:, fe2].min() - 5, norm[:, fe2].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
fig = plt.figure()
bx = fig.add_subplot( 111 )

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

bx.pcolormesh(xx, yy, Z, cmap=cmap_light)


##################################################################
#
# Multivariate normal boundary
#
##################################################################

X = norm[:, [fe1, fe2]]

chars = np.zeros((3, 10, 2))
for i in range(3):
    for j in range(10):
        chars[i][j][0] = X[j + (10*i)][0]
        chars[i][j][1] = X[j + (10*i)][1]

print(chars)
means = np.empty((3,2))
for i in range(3):
    means[i] = np.mean(chars[i], axis=0)
print(means)

covs = []
for i in range(3):
    covs.append(np.cov(chars[i].T))
print(covs)

delta = 0.1
x = np.arange(norm[:, fe1].min() - 5, norm[:, fe1].max() + 5 , delta)
y = np.arange(norm[:, fe2].min() - 5, norm[:, fe2].max() + 5 , delta)
X, Y = np.meshgrid(x,y)
pos = np.dstack((X, Y))

mv1 = multivariate_normal.pdf(pos, means[0], covs[0])
mv2 = multivariate_normal.pdf(pos, means[1], covs[1])
mv3 = multivariate_normal.pdf(pos, means[2], covs[2])

r1 = mv1 - mv2
r2 = mv1 - mv3
r3 = mv2 - mv3

plt.contour(X, Y, r1, 3, colors='blue')
plt.contour(X, Y, r2, 3, colors='red' )
plt.contour(X, Y, r3, 3, colors='green')


###################################################################
##
## Plotting characters
##
###################################################################

## Training characters
bx.scatter(norm[:, fe1], norm[:, fe2], c=targets, cmap=cmap_bold)

## Test characters
bx.scatter(normT[:, fe1], normT[:, fe2], c=targetsT, cmap=cmap_bold, marker="*", s=100)

## A character
bx.scatter(aNorm[fe1], aNorm[fe2], c="y", cmap=cmap_bold, marker="s", s=100) # A character

## B character
bx.scatter(bNorm[fe1], bNorm[fe2], c="y", cmap=cmap_bold, marker="D", s=100) # B character


plt.show()
