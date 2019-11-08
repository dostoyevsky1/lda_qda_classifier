"""
@author: mdrozdov -- dostoyevsky1
"""
### LDA/QDA Classifier

import numpy as np
from numpy import linalg
from sklearn.metrics import accuracy_score

class lda_qda_classifier(object):
    def __init__(self, cov_dict = None, probs_dict = None, 
                        mu_dict = None, classes_dict = None, QDA = False):
        self.cov_dict = cov_dict
        self.probs_dict = probs_dict
        self.mu_dict = mu_dict
        self.classes_dict = classes_dict
        self.QDA = QDA

    def get_classes(self, y):
        self.classes_dict = {}
        for num, item in enumerate(np.unique(y)):
            self.classes_dict[num] = item

    def get_probs(self, y):
        class_counts_dict = dict(zip(*np.unique(y,return_counts=True)))
        self.probs_dict = {}
        for key,value in self.classes_dict.items():
            self.probs_dict[key] = class_counts_dict[value]/y.shape[0]

    def get_means(self, X):
        self.mu_dict = {}
        for key, value in self.classes_dict.items():
            self.mu_dict[key] = np.mean(X[y==value], axis = 0)

    def get_cov(self, X):
        self.cov_dict = {}
        for key, value in self.classes_dict.items():
            self.cov_dict[key] = np.cov(X[y==value], rowvar = False)

    def fit(self, X, y):
        if self.classes_dict is None:
            self.get_classes(y)
        if self.probs_dict is None:
            self.get_probs(y)
        if self.cov_dict is None:
            self.get_cov(X)
        if self.mu_dict is None:
            self.get_means(X)
    
    def predict_class(self, X):
        self.score_list = []
        if self.QDA:
            for p in self.classes_dict.keys():
                score = list(map(lambda x: ((-0.5)*(x-self.mu_dict[p])).T.dot(linalg.inv(self.cov_dict[p])).dot(x-self.mu_dict[p])-(0.5)*np.log(linalg.det(self.cov_dict[p]))+np.log(self.probs_dict[p]), X))
                self.score_list.append(score)

        else:
            for p in self.classes_dict.keys():
                score = list(map(lambda x: x.T.dot(linalg.inv(self.cov_dict[0])).dot(self.mu_dict[p])-(0.5)*np.array(self.mu_dict[p]).T.dot(linalg.inv(self.cov_dict[0])).dot(self.mu_dict[p])+np.log(self.probs_dict[p]), X))
                self.score_list.append(score)

        self.score_list = np.array(list(zip(*self.score_list)))
        self.classified = np.argmax(self.score_list,axis=1)
    
    
                

# TESTING    
# Data Setup

# sigma = 0.5
# J = 3
# N = 100

# rho_dict = {}
# for i in range(1,J+1):
#     rho = 0.25 + (i-1)*0.25
#     rho_dict[i-1] = rho

# cov_dict = {}
# for i in range(J):
#     cov = np.array([[1,rho_dict[i]],[rho_dict[i],1]]) * (sigma**2)
#     cov_dict[i] = cov
    
# probs_dict = {0:0.6,1:0.3,2:0.1}

# mu1 = [0, 1]
# x1, y1 = np.random.multivariate_normal(mu1, cov_dict[0], int(N*probs_dict[0])).T

# mu2 = [1, 0]
# x2, y2 = np.random.multivariate_normal(mu2, cov_dict[1], int(N*probs_dict[1])).T

# mu3 = [-1, 1]
# x3, y3 = np.random.multivariate_normal(mu3, cov_dict[2], int(N*probs_dict[2])).T

# mu_dict = {0:mu1, 1:mu2, 2:mu3}


# # Generate Data
# X = np.asarray(np.vstack((np.hstack((x1,x2,x3)),np.hstack((y1,y2,y3)))).T)
# y = np.hstack((np.zeros(int(N*probs_dict[0])),np.ones(int(N*probs_dict[1])),np.ones(int(N*probs_dict[2]))*2))




# LDA = lda_qda_classifier(probs_dict = probs_dict, cov_dict=cov_dict, mu_dict = mu_dict)
# LDA.fit(X,y)
# print(LDA.classes_dict)
# print(LDA.probs_dict)
# print(LDA.cov_dict)
# print(LDA.mu_dict)
# LDA.predict_class(X)

# QDA = lda_qda_classifier(probs_dict = probs_dict, cov_dict = cov_dict,mu_dict = mu_dict,QDA=True)
# QDA.fit(X,y)
# QDA.predict_class(X)

# print(accuracy_score(y,LDA.classified))
# print(accuracy_score(y,QDA.classified))



