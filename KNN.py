'''
Created on Aug 29, 2016
This class is a decision tree implementation taken from Hal Daume.

@author: km_dh
'''
import numpy as np
from operator import itemgetter


def distance(p1, p2):
    return ((p1[1] - p2[1]) ** 2 + (p1[0]-p2[0]) ** 2)


class KNN(object):
        
    def res(self, mode='name', model=None, test_case=np.zeros(1), X=np.zeros(1), Y=np.zeros(1), cutoff=0):
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
            if(len(X) < 2 or len(Y) < 1 or cutoff < 1):
                print("Error: training requires three arguments: X, Y, and cutoff")
                return 0
            sizeX = X.shape
            sizeY = Y.shape
            if(sizeX[0] != sizeY[0]):
                print("Error: there must be the same number of data points in X and Y")
                return

            if len(sizeY) > 1 and sizeY[1] == 1:
                Y = np.reshape(Y, (sizeY[1], sizeY[0]))
                Y = Y[0]
                sizeY = Y.shape
            if(len(sizeY) != 1):
                print("Error: Y must have only 1 column")
                return 0
            if(cutoff not in range(1000)):
                print("Error: cutoff must be a positive scalar")
                return 0
            res = {'X': X, 'Y': Y, 'K': cutoff}
            return res
        
        if(mode == 'predict'):
            if(len(model) < 1 or len(test_case) < 1):
                print("Error: prediction requires two arguments: the model and X")
                return 0
            if('K' not in model.keys() and 'X' not in model.keys() and 'Y' not in model.keys()):
                print("Error: model does not appear to be a KNN model")
                return 0
            sizeModel = X.shape
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
        longest_dist = -1
        # Format is (Dist, Vote)
        nearest_neighbors = []
        accepting_neighbors = True
        for index, point in enumerate(model["X"]):
            dist = distance(point, test_case)
            if longest_dist > dist or accepting_neighbors:
                if len(nearest_neighbors) >= model["K"]:
                    # Retrieve the biggest distance currently, and then replace it
                    biggest_distance = max(nearest_neighbors, key=itemgetter(0))
                    nearest_neighbors[nearest_neighbors.index(biggest_distance)] = (dist, index)
                    # Re calculate our longest_distance
                    longest_dist = max(nearest_neighbors, key=itemgetter(0))[0]
                else:
                    # We can look up index later to save time
                    nearest_neighbors.append((dist, index))
                    # Stop adding new neighbors when we hit K and begin replacing
                    if len(nearest_neighbors) >= model["K"]:
                        accepting_neighbors = False
                        longest_dist = max(nearest_neighbors, key=itemgetter(0))[0]

        # tally our votes up
        votes = {}
        for neighbor in nearest_neighbors:
            vote = model["Y"][neighbor[1]]
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        winner = max(votes.iteritems(), key=itemgetter(1))[0]
        return winner
