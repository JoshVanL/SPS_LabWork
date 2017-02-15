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
#print(data)
#print(data.size)
#plt.hist(data, data.size, range=[np.amin(data)-0.5, np.amax(data)+0.5])
#plt.show()

def computeLikelihood(D, mu):
	return np.prod(2*stats.norm.pdf(4*((D-mu))**2))
	

num = computeLikelihood(data, 0)
print(num)

def loopLikelihood(D):
#	muList = np.array()
	muList = np.array([computeLikelihood(D, 0.0)])
	for i in np.arange(0.001, 1.00, 0.001):
		muList = np.concatenate((muList, [computeLikelihood(D, i)]))	
	print(muList)
	return muList

muList = loopLikelihood(data)
print(muList)
print('max p(D|mu) ', np.amax(muList))
muML = np.mean(data)
print('arg max mu P(D|mu) ', muML)
plt.plot(np.arange(0.0, 1.0, 0.001), muList)

def computeProstieror(D, mu):
	#print(2*stats.norm.pdf(4*((D-mu))**2))
	prior = stats.gamma.pdf(D, 0.5, 0.01)
	return computeLikelihood(D, mu) * prior

def loopPosterior(D):
	muList = np.array([computeProstieror(D, 0.0)])
	for i in np.arange(0.001, 1.00, 0.001):
		muList = np.concatenate((muList, [computeProstieror(D, i)]))	
	#print(muList)
	return muList

muMAP = loopPosterior(data)
#muMAP = np.max(muMAP)
print(muMAP)
plt.plot(np.arange(0.0, 1.0, 0.001), muMAP)
plt.show()
