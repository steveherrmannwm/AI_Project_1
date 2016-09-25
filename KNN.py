'''
Created on Aug 29, 2016
This class is a decision tree implementation taken from Hal Daume.

@author: km_dh
'''
import numpy as np
from operator import itemgetter


def distance(p1, p2):
    total_distance = -1
    if len(p1) == len(p2):
        total_distance = sum([(int(num_1) - int(num_2)) ** 2 for num_1, num_2 in zip(p1, p2)])
    return total_distance



class KNN(object):

    def res(self, mode='name', model=None, test_case=np.zeros(1), X=np.zeros(1), Y=np.zeros(1), h_param = 0):
        '''
        usage is of the two following:
        learn = KNN()
        model = learn.res('train', X=, Y=, K=)
        Y = learn.res('predict', model=, X=)
        '''
        if model is None:
            model = {}
        mode = mode.lower()

        if(mode == 'name'):
            return 'KNN'

        if(mode == 'train'):
            if(len(X) < 2 or len(Y) < 1 or h_param < 1):
                print("Error: training requires three arguments: X, Y, and cutoff")
                return 0
            sizeX = X.shape
            sizeY = Y.shape
            if(sizeX[0] != sizeY[0]):
                print("Error: there must be the same number of data points in X and Y")
                return 0
            if len(Y.shape) > 1 and Y.shape[1] == 1:
                Y = Y.flatten()
                sizeY = Y.shape

            if(len(sizeY) != 1):
                print("Error: Y must have only 1 column")
                return 0
            if(h_param not in range(1000)):
                print("Error: cutoff must be a positive scalar")
                return 0
            res = {'X': X, 'Y': Y, 'K': h_param}
            return res

        if(mode == 'predict'):
            if(len(model) < 1 or len(test_case) < 1):
                print("Error: prediction requires two arguments: the model and X")
                return 0
            if('K' not in model.keys() and 'X' not in model.keys() and 'Y' not in model.keys()):
                print("Error: model does not appear to be a KNN model")
                return 0
            sizeModel = model["X"].shape
            sizeX = test_case.shape
            if(len(sizeX) < 2):
                if(sizeModel[1] != sizeX[0]):
                    print("Error: there must be the same number of features in the model and X")
                res = self.KNNpredict(model, test_case)
            else:
                if(sizeModel[1] != sizeX[1]):
                    print("Error: there must be the same number of features in the model and X")
                N = sizeX[0]
                res = np.zeros(N)
                for n in range(N):
                    ans = self.KNNpredict(model, test_case[n,:])
                    res[n] = ans
            return res
        print("Error: unknown KNN mode: need train or predict")

    def KNNpredict(self, model, test_case):
        # model contains trainX which is NxD, trainY which is Nx1, K which is int. X is 1xD
        # We return a singe value 'y' which is the predicted class
        # print model
        # Set distance as negative, so we can replace later
        # This represents the furthest away of the nearest neighbors
        print "STARTING KNN"
        distances = sorted([(distance(model['X'][point], test_case), model['Y'][point]) for point in
                            xrange(len(model['X']))])

        votes = {label: 0 for label in model['Y']}
        # handle the rare case where we have fewer points than our expected K
        if model['K'] > len(distances):
            print "Reducing K to the number of points in the set, may want to add more data points"
            model['K'] = len(distances)

        for k in xrange(model['K']):
            votes[distances[k][1]] += 1
            print votes

        return sorted(votes.items(), key=itemgetter(1), reverse=True)[0][0]
