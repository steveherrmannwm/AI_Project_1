'''
Created on Aug 25, 2016
This class is a decision tree implementation taken from Hal Daume.

@author: km_dh
'''
import numpy as np
from collections import Counter


def most_common(lst):
    if lst.size > 0:
        return Counter(lst).most_common(1)[0][0]
    return 0.0


class DT(object):
    def __init__(self):
        self.completed_features = []

    def res(self, mode='name', model=None, test_case=np.zeros(1), X=np.zeros(1), Y=np.zeros(1), cutoff=-1):
        '''
        usage is of the two following:
        learn = DT()
        model = learn.res('train', X=, Y=, cutoff=)
        Y = learn.res('predict', model=, X=)
        '''
        if model is None:
            model = {}

        mode = mode.lower()

        if mode == 'name':
            return 'DT'

        if mode == 'train':
            if(len(X) < 2 or len(Y) < 1 or cutoff < 0):
                print("Error: training requires three arguments: X, Y")
                return 0
            sizeX = X.shape
            sizeY = Y.shape
            # WAS sizeX[0] - 1
            if sizeX[0]  != sizeY[0]:
                print("Error: there must be the same number of data points in X and Y")
                return 0
            if len(sizeY) > 1 and sizeY[1] == 1:
                Y = np.reshape(Y, (sizeY[1], sizeY[0]))
                Y = Y[0]
                sizeY = Y.shape

            return self.DTconstruct(X,Y,cutoff)

        if mode == 'predict':
            if len(model) < 1 or len(test_case) < 1:
                print("Error: prediction requires two arguments: the model and X")
                return 0
            if'isLeaf' not in model.keys():
                print("Error: model does not appear to be a DT model")
                return 0

            #set up output
            rowCol = test_case.shape
            if(len(rowCol) < 2):
                res = self.DTpredict(model, test_case)
            else:
                N = rowCol[0]
                res = np.zeros(N)
                for n in range(N):
                    ans = self.DTpredict(model, test_case[n,:])
                    res[n] = ans
            return res
        print("Error: unknown DT mode: need train or predict")

    def DTconstruct(self, X, Y, cutoff):
        # X COLUMNS ARE FEATURES
        # X ROWS ARE INDIVIDUAL DATA POINTS
        # Y IS WHAT EACH POINT SHOULD BE CLASSIFIED AS
        # Truncate our data to the cutoff if specified
        # Have a standard guess in case we hit a case to end
        guess = most_common(Y)
        # handle the case where all labels are the same
        print "STARTED BUILDING TREES"
        if len(set(Y)) == 1:
            return {"isLeaf": 1, "label": guess}

        # We've run out of features, so finish the tree with our best guess
        if len(self.completed_features) == X.shape[1]:
            return {"isLeaf": 1, "label": guess}

        # We've hit the cutoff, because we ignore cutoffs of 0 or below
        if cutoff == 1:
            return {"isLeaf": 1, "label": guess}

        # Tally up our votes, so we can chose the next feature to branch on
        max_votes = -1
        feature_to_check = -1

        columns_to_search = [x for x in xrange(X.shape[1]) if x not in self.completed_features]
        rows_to_search = X.shape[0]

        print columns_to_search
        print rows_to_search
        # Get the votes from each feature that hasn't been touched
        votes = {column:0 for column in columns_to_search}
        for row in xrange(rows_to_search):
            for column in columns_to_search:
                print row, column
                # Weight the algorithm to favor features which are easier to find discrepancies
                votes[column] += 1
                    # Append the row to the array horizontally
        print votes
        for key in votes:
            if votes[key] ** 2 > max_votes:
                max_votes = votes[key] ** 2
                feature_to_check = key
            # Square to remove negative sign

        self.completed_features.append(feature_to_check)
        # print feature_to_check
        column = np.swapaxes(X, 1, 0)[feature_to_check]
        rows_to_split = np.where(column >= 0.5)[0]

        yes_rows = np.array([X[row] for row in rows_to_split])
        yes_label_list = np.array([Y[row] for row in rows_to_split])
        no_rows = np.array([X[row] for row in xrange(len(column)) if row not in rows_to_split])
        no_label_list = np.array([Y[row] for row in xrange(len(column)) if row not in rows_to_split])

        yes_data = np.array([], dtype="int64").reshape((0, X.shape[1]))
        yes_labels = np.array([])
        no_data = np.array([], dtype="int64").reshape((0, X.shape[1]))
        no_labels = np.array([])

        yes_data = np.concatenate((yes_data, yes_rows), axis=0)
        yes_labels = np.concatenate((yes_labels, yes_label_list))
        no_data = np.concatenate((no_data, no_rows), axis=0)
        no_labels = np.concatenate((no_labels, no_label_list))

        # Build our node, and set off the left and right nodes
        print "building our next tree"
        tree = {'isLeaf': 0, 'split': feature_to_check,
                'left': self.DTconstruct(X=no_data, Y=no_labels, cutoff=(cutoff - 1)),
                'right': self.DTconstruct(X=yes_data, Y=yes_labels, cutoff=(cutoff - 1))}

        return tree

        # the Data comes in as X which is NxD and Y which is Nx1.
        # cutoff is a scalar value. We should stop splitting when N is <= cutoff
        #
        # features (X) may not be binary... you should *threshold* them at
        # 0.5, so that anything < 0.5 is a "0" and anything >= 0.5 is a "1"
        #
        # we want to return a *tree*. the way we represent this in our model
        # is that the tree is a Python dictionary.
        #
        # to represent a *leaf* that predicts class 3, we can say:
        #    tree = {}
        #    tree['isLeaf'] = 1
        #    tree['label'] = 3
        #
        # to represent a split node, where we split on feature 5 and then
        # if feature 5 has value 0, we go down the left tree and if feature 5
        # has value 1, we go down the right tree.
        #    tree = {}
        #    tree['isLeaf'] = 0
        #    tree['split'] = 5
        #    tree['left'] = ...some other tree...
        #    tree['right'] = ...some other tree...

    def DTpredict(self,model,X):
        # here we get a tree (in the same format as for DTconstruct) and
        # a single 1xD example that we need to predict with
        if model['isLeaf'] == 1:
            return model['label']

        if X[model['split']] < 0.5:
            return self.DTpredict(model['left'], X)

        return self.DTpredict(model['right'], X)

    def DTdraw(self,model, level=0):
        indent = ' '
        if model is None:
            return
        # print indent*4*level + 'isLeaf: ' + str(model['isLeaf']) + "|" + level
        if model['isLeaf']==1:
            print indent*4*level + 'Y: ' + str(model['label']) + "|" + str(level)
            return
        print indent*4*level + 'split ' + str(model['split']) + "|" + str(level)
        left_tree = str(self.DTdraw(model['left'], level+1))
        if left_tree != 'None':
            #print model['left']
            print indent*4*level + 'left: ' + left_tree + "|" + str(level)
        right_tree = str(self.DTdraw(model['right'],level+1))
        if right_tree != 'None':
            #print model['right']
            print indent*4*level + 'right: ' + right_tree + "|" + str(level)
