
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

#%matplotlib inline



def plotmatrix(Matrix):
    r, c   = Matrix.shape
    fig    = plt.figure()
    plotID = 1
    for i in range(c):
        for j in range(c):
            ax = fig.add_subplot( c, c, plotID )
            ax.scatter( Matrix[:,i], Matrix[:,j] )
            plotID += 1
    print('about to show')
    plt.show()



#generates fourier space
def read(char_name):
    f      = io.imread(char_name)   # read in image
    f_f    = np.array(f, dtype=float)
    z      = np.fft.fft2(f_f)           # do fourier transform
    q      = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq   =  np.absolute(q)         # magnitude spectrum
    Phaseq = np.angle(q)           # phase spectrum
    # return Magq
    return q



#fourier's conjugate symmetry => no need to consider the bottom of a char's f.space
def slice(old_matrix):
    factor     = old_matrix.shape[0]/2
    new_matrix = old_matrix[:factor,:]
    return new_matrix



#generates a matrix of magnitudes of all characters
def gen_train_data():
    # mag = np.zeros((200,640))
    q = np.zeros((200,640))
    for c in ['V','T','S']:
        for i in range(1,11):
            # mag = np.concatenate((mag, slice(read("chars/" + c + str(i) + ".GIF"))), axis = 0)
            q = np.concatenate((q, slice(read("chars/" + c + str(i) + ".GIF"))), axis = 0)

    # mag = mag[200:,:] #getting rid of the zero rows added at the beg of this function
    # return mag
    q = q[200:,:] #getting rid of the zero rows added at the beg of this function
    return q



#splits a matrix of all characters into "seperate" characters
#e.g mags[0] might denote char 'S', m[1] - 'V', m[2] - 'V'
def split(matrix):
    char_type    = matrix.shape[0]
    step         = round(matrix.shape[0] / 3)
    # mags         = [[] for x in range(char_type)]
    qs           = [[] for x in range(char_type)]
    for i,s in zip(range(char_type), range(0, char_type, step)):
        # mags[i]  = matrix[s:(s+step)]
        qs[i]    = matrix[s:(s+step)]
    # return mags
    return qs


def spectralRegion(data):
    feature1 = []
    feature2 = []
    center = (round(data.shape[0]/2), round(data.shape[1]/2))
    for j in range (data.shape[1]):
        for i in range(data.shape[0]):
            mag = np.absolute(data[i][j])
            if (mag != 0):
                n = center[0] - i
                m = center[1] - j
                if (m != 0):
                    if (( math.tan(n/m) >= (3*math.pi/8) and  math.tan(n/m) <= (math.pi/2)) or
                        ( math.tan(n/m) <=    math.pi/8)):
                            feature1.append(mag)

                    if ( (math.tan(n/m) <= (3*math.pi/8)) and  math.tan(n/m) >= (math.pi/8)):
                            feature2.append(mag)

                if (n != 0):
                    if ( (math.tan(m/n) >= (3*math.pi/8) and  math.tan(m/n) <= (math.pi/2)) or
                         (math.tan(m/n) <=    math.pi/8)):
                            feature1.append(mag)

                    if (( math.tan(m/n) <= (3*math.pi/8)) and  math.tan(m/n) >= (math.pi/8)):
                            feature2.append(mag)

    f = list(map(lambda x, y: [x,y], feature1, feature2))
    features = np.matrix(f)
    return features


data = gen_train_data()
data_split = split(data)

#getting feature matrix for each char
features        = [[] for x in range(3)]
mfeatures       = np.matrix(np.zeros((1,2)))
for char,i in zip(data_split, range(3)):
    features[i] = spectralRegion(char)
    mfeatures   = np.concatenate((mfeatures, features[i]), axis=0)

def normalise(matrix):
    col1 = list(itertools.chain(*(matrix[:,0].tolist())))
    col2 = list(itertools.chain(*(matrix[:,1].tolist())))
    col1_norm = col1/ np.linalg.norm(col1)
    col2_norm = col2/ np.linalg.norm(col2)
    m = [[],[]]
    m[0] = col1_norm
    m[1] = col2_norm
    return np.matrix(m).T


y = list(map(lambda x, y: [x]*y, [0,1,2], [features[0].shape[0], features[1].shape[0], features[2].shape[0]]))
mfeatures = mfeatures[1:,:]
labels = list(itertools.chain(*y))



n_neighbors = 15

# import some data to play with
#X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
#y = k[1]
#
h = 10000  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', n_jobs=-1)
clf.fit(mfeatures, labels)
#clf = NearestNeighbors(n_neighbors=15)
#clf.fit(mfeatures)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = mfeatures[:, 0].min() - 1, (mfeatures[:, 0].max() + 1)
y_min, y_max = mfeatures[:, 1].min() - 1, (mfeatures[:, 1].max() + 1)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
#Z = neigh.kneighbors_graph(mfeatures)
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

Plot also the training points
plt.scatter(features[:, 0], features[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')"
         % (n_neighbors, 'distance'))

plt.show()
