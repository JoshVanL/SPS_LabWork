
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
    r, c = Matrix.shape
    fig = plt.figure()
    plotID = 1
    for i in range(c):
        for j in range(c):
            ax = fig.add_subplot( c, c, plotID )
            ax.scatter( Matrix[:,i], Matrix[:,j] )
            plotID += 1
    plt.show()



def read(char_name):
    f = io.imread(char_name)   # read in image
    f_f = np.array(f, dtype=float)
    z = np.fft.fft2(f_f)           # do fourier transform
    q = np.fft.fftshift(z)         # puts u=0,v=0 in the centre
    Magq =  np.absolute(q)         # magnitude spectrum
    Phaseq = np.angle(q)           # phase spectrum
    return Magq



#fourier's conjugate symmetry => no need to consider the bottom of a char's f.space
def slice(old_matrix):
    factor = old_matrix.shape[0]/2
    new_matrix = old_matrix[:factor,:]
    return new_matrix



print(slice(read("chars/V1.GIF")).shape)
#shape is (400,640) - 400 rows , 640 columns
#mac shows that image dimensions are (640,400) 640 x-values(columns)m 400 y-values(rows)
# since a forier space is being analyzed, there is no need to look at the whole image for the forier conjugate symmetry.
# this implies, the forier image can be cut in half and only one part needs to be analysed e,g. only first 200 rows


# for i in range(1, 11):
#     read("chars/V"+i+".GIF")
