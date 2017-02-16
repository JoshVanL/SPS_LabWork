import numpy as np
from scipy import stats
from skimage import data, io, color, transform, exposure
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%matplotlib inline
# notebook
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] =  (32.0, 24.0)
pylab.rcParams['font.size'] = 24


fig = plt.figure()
ax = fig.add_subplot( 111  )
data = np.loadtxt('data1.dat')

def computeLikelihood(D, mu):
	return np.prod(stats.norm.pdf(D, mu, 0.5))


num = computeLikelihood(data, 0)

def loopLikelihood(D):
	muList = np.array([computeLikelihood(D, 0.0)])
	for i in np.arange(0.001, 1.00, 0.001):
		muList = np.concatenate((muList, [computeLikelihood(D, i)]))	
	return muList

muList = loopLikelihood(data)
print('max p(D|mu) ', np.amax(muList))
muML = np.mean(data)
print('arg max mu P(D|mu) ', muML)
pl = plt.plot(np.arange(0.0, 1.0, 0.001), muList, 'r--')

def computeProstieror(D, mu):
	prior = stats.norm(0.5, 0.01**0.5).pdf(mu)
	return computeLikelihood(D, mu)*prior

def loopPosterior(D):
	posList = np.array([computeProstieror(D, 0.0)])
	for i in np.arange(0.001, 1.00, 0.001):
		posList = np.concatenate((posList, [computeProstieror(D, i)]))	
	return posList

muMAP = loopPosterior(data)
pl = plt.plot(np.arange(0.0, 1.0, 0.001), muMAP)
plt.ylabel('probability')
plt.xlabel('theta')
plt.show()
