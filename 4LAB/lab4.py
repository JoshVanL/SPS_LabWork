import numpy as np
from scipy import stats



def computeLikelihood(D, mu):
	return np.prod(stats.norm.pdf(D, mu, 0.5))


def loopLikelihood(D):
	muList = np.array([computeLikelihood(D, 0.0)])
	for i in np.arange(0.001, 1.00, 0.001):
		muList = np.concatenate((muList, [computeLikelihood(D, i)]))	
	return muList

def computeProstieror(D, mu):
	prior = stats.norm(0.5, 0.01**0.5).pdf(mu)
	return computeLikelihood(D, mu)*prior

def loopPosterior(D):
	posList = np.array([computeProstieror(D, 0.0)])
	for i in np.arange(0.001, 1.00, 0.001):
		posList = np.concatenate((posList, [computeProstieror(D, i)]))	
	return posList


