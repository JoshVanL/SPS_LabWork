from __future__ import print_function
from scipy import stats
from skimage import data, io, color, transform, exposure
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (32.0, 24.0)
pylab.rcParams['font.size'] = 24

## als = (XtX)^-1 XtY
def getAls (x,y,d):
    Y = np.array(y).reshape(x.size, 1)
    X = np.arange(float(x.size*d)).reshape(x.size, d)
    X[:,0] = 1
    for i in range(0, x.size):
        for j in range(1, d):
            X[i][j] = x[i]**j

    X = np.matrix(X)
    Xt = X.transpose()
    als = np.linalg.inv(Xt*X)*Xt*Y
    return als

print("hello")
csv = np.genfromtxt('DMD.csv', delimiter=',')
x = csv[:,0]
y = csv[:,1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x,y)

als = getAls(x,y,10)
xp = np.arange(-1.5, 11, 0.01)
yp = np.array(als.item(0) + als.item(1)*xp + als.item(2)*xp**2
             + als.item(3)*xp**3 + als.item(4)*xp**4
             + als.item(5)*xp**5 + als.item(6)*xp**6
             + als.item(7)*xp**7 + als.item(8)*xp**8
             + als.item(9)*xp**9)
pl = plt.plot(xp,yp)

als = getAls(x,y,2)
xp = np.arange(-1.5, 11, 0.01)
yp = np.array(als.item(0) + als.item(1)*xp)
pl = plt.plot(xp,yp)
