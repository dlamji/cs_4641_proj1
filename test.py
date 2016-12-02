import numpy as np

filePath = "data/univariateData.dat"
file = open(filePath,'r')
allData = np.loadtxt(file,delimiter=',')
X = np.matrix(allData[:,:-1])
y = np.matrix(allData[:,-1]).T
n,d = X.shape


theta = [[1,2],[4,3]]

x1 = [3,4,5]
x2 = [1,2,3]
x3 = [[2,3],[6,7]]

theta = np.asarray(theta)
oldtheta = theta
oldtheta = oldtheta.T
x1 = np.asarray(x1)
x2 = np.asarray(x2)

#print theta.dot(x)
#print theta * x
print x1.T
print x2
print oldtheta
print theta
print np.linalg.inv(theta.T.dot(x3).dot(x3)).dot(x3)
print x1.dot(x2)