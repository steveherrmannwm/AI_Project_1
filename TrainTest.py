'''
Created on Aug 28, 2016

@author: km_dh
'''
import numpy as np
import time

class TrainTest(object):
    '''
    classdocs
    '''

    def __init__(self, learn, trainX=np.array([[]]), trainY=np.array([]), testX=np.array([[]]), testY=np.array([]),
                 h_param=0):
        self.learn = learn
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.h_param = h_param

    def run_tt(self):
        output = {}
        print 'Training...'
        t0 = time.time()
        model = self.learn.res('train', X=self.trainX, Y=self.trainY, h_param=self.h_param)
        if not isinstance(model, dict) or model is None:
            print "model did not train"
            return 0
        output['model'] = model
        t1 = time.time()
        output['trainTime'] = t1-t0
        print 'Testing...'
        t0 = time.time()
        Y = self.learn.res('predict', model=model, test_case=self.testX)
        t1 = time.time()
        output['testTime'] = t1 - t0
        Y = np.array(Y)
        sizeY = Y.shape
        if len(sizeY) < 2:
            Y = np.reshape(Y, (sizeY[0],1))

        if(len(Y) == len(self.testY)):
            overlp = [Y == self.testY]
            overlp_sum = sum(sum(overlp))
            output['acc'] = overlp_sum/(len(Y)*1.0)
            return output
        print 'cannot determine accuracy'
        return 0

    def verifyAcc(self,acc, desired):
        print "accuracy is ", acc, "... should be above ", desired, "..."
        if acc >= desired:
            print "passed!"
            return 1
        print "failed"
        return 0


