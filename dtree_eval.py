'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # initialize accuracy list
    trainAccuracy = []
    trainStumpAccuracy = []
    trainDT3Accuracy = []

    # totally 100 trials
    for i in range (100):
        # shuffle the data for each trial
    	idx = np.arange(n)
    	np.random.seed(13)
    	np.random.shuffle(idx)
    	X = X[idx]
    	y = y[idx]
        # 10 folds each trial
        for i in range (10):
            # make it a list, easy for deleting data
            xtrain = X.tolist()
            xtest = []
            ytrain = y.tolist()
            ytest = []

            # index of testing data for each fold
            # max list number is [266] so should be 267*i/10
            idxS = int(267*i/10)
            idxE = int(267*(i+1)/10)

            # separate training data and practice data
            for i in range(idxS,idxE):
                xtest.append(xtrain[i][:])
                ytest.append(ytrain[i][:])
                del xtrain[i][:]
                del ytrain[i][:]

            # remove deleted empty lists in order to fit into the function
            xtrain = np.asarray([x for x in xtrain if x!=[]]); 
            ytrain = np.asarray([x for x in ytrain if x!=[]]);   

            # train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(xtrain,ytrain)
            y_pred = clf.predict(xtest)
            trainAccuracy.append(accuracy_score(ytest, y_pred))

            clfStump = tree.DecisionTreeClassifier(max_depth = 1)
            clfStump = clfStump.fit(xtrain,ytrain)
            ystump_pred = clfStump.predict(xtest)
            trainStumpAccuracy.append(accuracy_score(ytest, ystump_pred))

            clfDT3 = tree.DecisionTreeClassifier(max_depth = 3)
            clfDT3 = clfDT3.fit(xtrain,ytrain)
            yDT3_pred = clfDT3.predict(xtest)
            trainDT3Accuracy.append(accuracy_score(ytest, yDT3_pred))


    # TODO: update these statistics based on the results of your experiment
    trainAccuracy = np.asarray(trainAccuracy)
    trainStumpAccuracy = np.asarray(trainStumpAccuracy)
    trainDT3Accuracy = np.asarray(trainDT3Accuracy)

    meanDecisionTreeAccuracy = np.mean(trainAccuracy)
    stddevDecisionTreeAccuracy = np.std(trainAccuracy)

    meanDecisionStumpAccuracy = np.mean(trainStumpAccuracy)
    stddevDecisionStumpAccuracy = np.std(trainStumpAccuracy)

    meanDT3Accuracy = np.mean(trainDT3Accuracy)
    stddevDT3Accuracy = np.std(trainDT3Accuracy)


    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
