import lab4
import numpy as np
from scipy import stats
from skimage import data, io, color, transform, exposure
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot( 111  )
data = np.loadtxt('data1.dat')


num = lab4.computeLikelihood(data, 0)


muList = lab4.loopLikelihood(data)
print('max p(D|mu) ', np.argmax(muList))
muML = np.mean(data)
print('arg max mu P(D|mu) ', muML)
pl = plt.plot(np.arange(0.0, 1.0, 0.001), muList, 'r--')

muMAP = lab4.loopPosterior(data)
pl = plt.plot(np.arange(0.0, 1.0, 0.001), muMAP, 'r')

data2 = np.loadtxt('data2.dat')
muList2 = lab4.loopLikelihood(data2)
pl2 = plt.plot(np.arange(0.0, 1.0, 0.001), muList2, 'b--')
muMAP2 = lab4.loopPosterior(data2)
pl2 = plt.plot(np.arange(0.0, 1.0, 0.001), muMAP2, 'b')

data3 = np.loadtxt('data3.dat')
muList3 = lab4.loopLikelihood(data3)
pl3 = plt.plot(np.arange(0.0, 1.0, 0.001), muList3, 'g--')
muMAP3 = lab4.loopPosterior(data3)
pl3 = plt.plot(np.arange(0.0, 1.0, 0.001), muMAP3, 'g')
plt.ylabel('probability')
plt.xlabel('theta')
plt.show()
