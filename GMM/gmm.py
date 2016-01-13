#!/usr/bin/env python
#coding:utf-8
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Normal(object):
    '''
    A class for store the parameter of normal distribution.
    '''

    def __init__(self, dim, mu = None, sigma = None):
        '''
        :param dim: int
                    The dimensions of the feature.
        :param mu: array optional
                    The mean of the data.
        :param sigma: array optional
                    The covariance of the data.
        '''

        self.dim = dim
        if not mu is None and not sigma is None:
            mu = mu
            sigma = sigma
        else:
            mu = np.random.randn(dim)
            sigma = np.eyes(dim, dtype = 'double')

        self.update(mu, sigma)

    def __str__(self):
        return "%s, %s" % (str(self.mu), str(self.sigma))

    def getMu(self):
        return self.mu

    def getSigma(self):
        return self.sigma

    def update(self, mu, sigma):
        '''
        Update the gaussian distribution parameter
        :param mu: array
                    The new mean vector.
        :param sigma: array
                    The new covariance matrix.
        :param A: array
                    The inverse of sigma.
        :param factor: double
                    The value of (2*pi)^(2/dim)abs(sigma)^0.5
        '''

        self.mu = mu
        self.sigma = sigma

        det = None
        if self.dim == 1:
            self.A = 1. / self.sigma
            det = np.fabs(self.sigma)
        else:
            self.A = np.inv(self.sigma)
            det = np.fabs(np.linalg.det(self.sigma))
        self.factor = (2.0*np.pi)**(0.5*self.dim)*(det)**(0.5)


    def pdf(self, x):
        '''
        Compute the vlaue of gaussian distribution
        '''

        A = self.A
        dx = x - self.mu
        res = np.exp(-0.5*np.dot(np.dot(dx, A), dx)) / self.factor
        return res

class GMM(object):
    '''
    A class for gaussian mixture model.
    '''
    def __init__(self, data, dim, group, params = None):
        '''

        :param data: array
                    The samples.
        :param dim: int
                    The dimension of the feature.
        :param group: int
                    The number of gaussian distribution.
        :param params: dict optional
                    The input parameter
        '''

        self.data = data
        if not params is None:
            self.dim = params['dim']
            self.group = params['group']
            self.priors = parmas['priors']
            self.params = params['parameter']
        else:
            self.dim = dim
            self.group = group
            self.params = []
            self.estimat()

    def __str__(self):

        res = "The group of %d\n" % self.group
        res += "%s\n" % str(self.priors)
        for i in range(self.group):
            res += "%s\n" % str(self.params[i])
        return res

    def estimat(self):
        '''
        Estimating the mean and covariance of the data.
        '''
        k = self.group
        n = len(self.data)
        data = self.data
        centroids = random.sample(self.data, k)
        clusters = [[] for i in range(k)]
        for x in data:
            i = np.argmin([np.linalg.norm(x - y) for y in centroids])
            clusters[i].append(x)

        for i in range(k):
            self.params.append(Normal(self.dim, mu = centroids[i], sigma = np.cov(clusters[i], rowvar = 0)))
        self.priors = [len(x) for x in clusters] / (np.ones(k)*n)

    def isEqual(self, cur, now):
        leng = len(cur)
        for i in range(leng):
            if abs(cur[i] - now[i]) > 0.000001:
                return False
        return True

    def em(self, iters = 100):
        '''

        :param iters: int
                    The max iters of the EM algorithm.
        :return:
        '''

        k = self.group
        d = self.dim
        n = len(self.data)
        data = self.data
        response = np.zeros((k, n))
        count = 0
        cur = copy.copy(self.priors)
        while iters > 0:

            # E-step

            for j in range(n):
                for i in range(k):
                    response[i,j] = self.params[i].pdf(data[j]) * self.priors[i]
            response /= np.sum(response, axis = 0)
            Nr = np.sum(response, axis = 1)

            # M-step

            for i in range(k):
                mu = np.dot(response[i,:], data) / Nr[i]
                sigma = np.zeros((d,d))
                for j in range(n):
                    sigma += response[i,j] * np.outer(data[j,:]-mu, data[j,:]-mu)
                sigma /= Nr[i]
                self.priors[i] = Nr[i] / n
                self.params[i].update(mu, sigma)
            iters -= 1
            count += 1

            #print the result;

            print "the %d iter: " %count
            print 'the priors: ', self.priors
            print 'the mu and sigma'
            for x in self.params:
                print x

            now = copy.copy(self.priors)
            if self.isEqual(now, cur):
                break
            cur = now[:]

        return (self.priors, self.params)


