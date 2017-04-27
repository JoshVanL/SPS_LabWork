
from   __future__             import print_function
from   sklearn.cluster        import KMeans
from   scipy.spatial.distance import cdist
from   scipy.spatial          import Voronoi, voronoi_plot_2d
from   scipy                  import stats
from   skimage                import io
import numpy                  as np
import matplotlib.pyplot      as plt
import itertools

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
    return Magq



#fourier's conjugate symmetry => no need to consider the bottom of a char's f.space
def slice(old_matrix):
    factor     = old_matrix.shape[0]/2
    new_matrix = old_matrix[:factor,:]
    return new_matrix



#generates a matrix of magnitudes of all characters
def gen_train_data():
    mag = np.zeros((200,640))
    for c in ['V','T','S']:
        for i in range(1,11):
            mag = np.concatenate((mag, slice(read("chars/" + c + str(i) + ".GIF"))), axis = 0)

    mag = mag[200:,:] #getting rid of the zero rows added at the beg of this function
    return mag



#splits a matrix of all characters into "seperate" characters
#e.g mags[0] might denote char 'S', m[1] - 'V', m[2] - 'V'
def split(matrix):
    char_type    = matrix.shape[0]
    step         = round(matrix.shape[0] / 3)
    mags         = [[] for x in range(char_type)]
    for i,s in zip(range(char_type), range(0, char_type, step)):
        mags[i]  = matrix[s:(s+step)]
    return mags


data = gen_train_data()
print(data.shape)
data_split = split(data)
print(data_split[0].shape)

#shape is (400,640) - 400 rows , 640 columns
#mac shows that image dimensions are (640,400) 640 x-values(columns)m 400 y-values(rows)
# since a forier space is being analyzed, there is no need to look at the whole image for the forier conjugate symmetry.
# this implies, the forier image can be cut in half and only one part needs to be analysed e,g. only first 200 rows
