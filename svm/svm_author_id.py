#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(C=10000.0,kernel='rbf')

t0 = time()
clf.fit(features_train,labels_train)
print "training time ", time() - t0


t1 = time()
pred = clf.predict(features_test)
print "prediction time ", time() - t1

score = accuracy_score(pred,labels_test)

count = 0
for i in pred:
    if i ==1:
        count = count + 1
print count
#print len(labels_test)
print len(labels_train)
print score



