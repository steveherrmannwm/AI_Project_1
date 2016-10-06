'''
Created on Aug 25, 2016
This class is a decision tree implementation taken from Hal Daume.

@author: km_dh
'''
import numpy as np
from collections import Counter
import math


def most_common(lst):
    if len(lst) > 0:
        return Counter(lst).most_common(1)[0][0]
    return 0.0


class DT(object):

    def res(self, mode='name', model=None, test_case=np.zeros(1), X=np.zeros(1), Y=np.zeros(1), h_param=-1):
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
            if(len(X) < 2 or len(Y) < 1 or h_param < 0):
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
            return self.DTconstruct(X, Y, h_param)

        if mode == 'predict':
            if len(model) < 1 or len(test_case) < 1:
                print("Error: prediction requires two arguments: the model and X")
                return 0
            if'isLeaf' not in model.keys():
                print("Error: model does not appear to be a DT model")
                return 0

            # set up output
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

        if len(set(Y)) == 1 or 0 in X.shape or cutoff == 1:
            return {"isLeaf": 1, "label": guess}
        # Find what feature we should select next
        columns_to_search = xrange(X.shape[1])
        rows_to_search = X.shape[0]
        #create the dict for holding the label counter dict once so we dont have to keep creating it
        labels_count_dict = {label: 0.0 for label in Y}
        votes = {feature: {"yes": labels_count_dict.copy(), "no": labels_count_dict.copy()} for feature in
                 columns_to_search}

        # Get the votes from each feature that hasn't been touched
        for row in xrange(rows_to_search):
            label = Y[row]
            for column in columns_to_search:
                # Weight the algorithm to favor features which are easier to find discrepancies
                if X[row][column] >= 0.5:
                    votes[column]["yes"][label] += 1
                else:
                    votes[column]["no"][label] += 1

                    # Append the row to the array horizontally
        # To calculate majority vote, take the label with the highest data gain from the yes data, and add it to the
        # label with the highest no data
        # Square to remove negative sign
        label_counts = {label: 0.0 for label in Y}

        # simple way to get around filtering zeros out of np.bincount
        # count up how many times each label occurs
        for label in Y:
            label_counts[label] += 1

        set_entropy = []
        # if a label is here it has to have a non-zero value
        for label in label_counts:
            proportion = label_counts[label]/len(Y)
            set_entropy.append(-proportion * math.log(proportion, len(Y)))

        set_entropy = sum(set_entropy)
        feature_to_check = -1
        best_entropy = -1
        for feature in votes:
            feature_entropy = []
            for split in votes[feature]:
                branch_entropy = []
                label_total = sum(votes[feature][split].values())
                label_total = label_total if label_total > 0 else 1
                for label in votes[feature][split]:
                    proportion = votes[feature][split][label] / label_total
                    proportion = proportion if proportion > 0 else 1
                    entropy = -proportion * math.log(proportion, len(Y))
                    branch_entropy.append(label_total/len(Y) * entropy)
                branch_entropy = sum(branch_entropy)
                feature_entropy.append(branch_entropy)
            feature_entropy = sum(feature_entropy)
            if set_entropy - feature_entropy > best_entropy:
                best_entropy = set_entropy - feature_entropy
                feature_to_check = feature

        column = np.swapaxes(X, 1, 0)[feature_to_check]
        rows_to_split = np.where(column >= 0.5)[0]

        yes_rows = np.array([X[row] for row in rows_to_split])
        yes_label_list = np.array([Y[row] for row in rows_to_split])
        no_rows = np.array([X[row] for row in xrange(len(column)) if row not in rows_to_split])
        no_label_list = np.array([Y[row] for row in xrange(len(column)) if row not in rows_to_split])

        yes_data = np.array([], dtype="int8").reshape((0, X.shape[1]))
        yes_labels = np.array([])
        no_data = np.array([], dtype="int8").reshape((0, X.shape[1]))
        no_labels = np.array([])
        np.where(X.swapaxes(1, 0)[feature_to_check])

        if len(yes_rows) > 0:
            yes_data = np.concatenate((yes_data, yes_rows), axis=0)
            yes_labels = np.concatenate((yes_labels, yes_label_list))
        if len(no_rows) > 0:
            no_data = np.concatenate((no_data, no_rows), axis=0)
            no_labels = np.concatenate((no_labels, no_label_list))

        yes_data = np.delete(yes_data, feature_to_check, 1)
        no_data = np.delete(no_data, feature_to_check, 1)

        # Handle the case where we have no more data in one of the groups, and should stop splitting
        if 0 in yes_data.shape or 0 in no_data.shape:
            if 0 in yes_data:
                best_label = most_common(no_labels)
                second_label = most_common([label for label in no_labels if label != best_label])
                right_tree = {"isLeaf": 1, 'label': second_label}
                left_tree = {"isLeaf": 1, 'label': best_label}
            else:
                best_label = most_common(yes_labels)
                second_label = most_common([label for label in yes_labels if label != best_label])
                right_tree = {"isLeaf": 1, 'label': best_label}
                left_tree = {"isLeaf": 1, 'label': second_label}
        else:
            # Build our node, and set off the left and right nodes
            right_tree = self.DTconstruct(X=yes_data, Y=yes_labels, cutoff=(cutoff - 1))
            left_tree = self.DTconstruct(X=no_data, Y=no_labels, cutoff=(cutoff - 1))

        tree = {'isLeaf': 0, 'split': feature_to_check,
                'left': left_tree, 'right': right_tree}

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

        if X[model['split']] >= 0.5:
            return self.DTpredict(model['left'], X)

        return self.DTpredict(model['right'], X)

    def DTdraw(self,model,level=0):
        indent = ' '
        if model is None:
            return
        print indent*4*level + 'isLeaf: ' + str(model['isLeaf']) + "|" + str(level)
        if model['isLeaf'] == 1:
            return indent*4*level + 'Y: ' + str(model['label']) + "|" + str(level)
        print indent*4*level + 'split ' + str(model['split']) + "|" + str(level)
        left_tree = str(self.DTdraw(model['left'],level+1))
        if left_tree != 'None':
            print indent*4*level + left_tree + "|" + str(level)
        right_tree = str(self.DTdraw(model['right'], level+1))
        if right_tree != 'None':
            print indent*4*level +  right_tree + "|" + str(level)