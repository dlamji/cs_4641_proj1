'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in xrange(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print "Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta
            
            # TODO:  add update equation here

            # this code is for 2.1 ~ 2.3
            #oldtheta = copy.deepcopy(theta) # make sure all thetas update at once

            #for eachtheta in range(len(oldtheta)):                
            #    sumoferror = 0.0
            #    for i in range(len(X)):
            #        sumoferror += float((theta.T.dot(X[i].T)-y[i])*X[i,eachtheta])                
            #   oldtheta[eachtheta] = theta[eachtheta] - (self.alpha/len(X))*sumoferror

            #theta = oldtheta # update theta at once

            # this code is for 2.4
            theta = theta - self.alpha*X.T.dot(X.dot(theta)-y)/n
              

        return theta
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        # TODO: add objective (cost) equation here
        sumoferror = 0.0
        for i in range(len(X)):
            sumoferror += (theta.T.dot(X[i].T) - y[i]) ** 2
        return float (sumoferror / (2 * len(X)))


    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        if self.theta==None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # TODO:  add prediction function here
        y = X * self.theta
        return y
