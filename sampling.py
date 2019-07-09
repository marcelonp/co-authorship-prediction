import os
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, OneClassSVM

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

TRAIN = pd.read_csv(ABS_PATH + "/train_.csv")
TEST = pd.read_csv(ABS_PATH + "/test_.csv")
BASE = pd.concat([TRAIN,TEST],ignore_index=True)
BASE = BASE.drop(columns=['CommonKeywords','CommonKeywords_this',"GeodesicDistance_Weighted",
                        "GeodesicDistance_Unweighted","Bigger_Component_Size","Smaller_Component_Size","Author1_Clustering","Author2_Clustering"])

BASE = BASE[BASE.columns[4:]]
print("antes:",len(BASE))
FILTRO = BASE.loc[~(BASE==0).all(axis=1)]
print("depois",len(FILTRO))

########################################################
#MÓDULO PARA TESTES COM AMOSTRAGEM
########################################################

class Sampling():

    def __init__(self):
        self.train = TRAIN.drop(columns=['CommonKeywords','CommonKeywords_this'])
        self.X_train = self.train[self.train.columns[5:]]
        self.y_train = self.train['Class']

        test = TEST.drop(columns=['CommonKeywords','CommonKeywords_this'])
        self.X_test = test[test.columns[5:]]
        self.y_test = test['Class']  

    def get_dic_instances(self, X_train, y_train, multiplier):
        """
        Gets de number of samples based on multiplier for the majority class
            param multiplier, float [0,1]
        """
        print("Getting instances... multiplier:", multiplier)
        dic_instances = Counter(y_train)
        dic_instances[0] = int(dic_instances[0]*multiplier)
        print(dic_instances)
        return (dic_instances)


    def undersample(self, X_train, y_train,dic_instances='auto'):
        print("Sampling...")
        rus = RandomUnderSampler(sampling_strategy=dic_instances)
        X_res,y_res = rus.fit_resample(X_train, y_train)
        return (X_res,y_res)

    def controller(self):
        #for i in np.arange(0.1, 1.0, 0.1):
        X_res,y_res = self.undersample(self.X_train, self.y_train,{0:10000,1:300})
        sns.pairplot(self.train)
        sns.plt.show()
        predicted = self.classify_dataset(X_res,y_res, self.X_test, self.y_test)
        self.save_classification_score(self.y_test,predicted)


        return
        
    def classify_dataset(self, X_res,y_res,X_test,y_test):
        clf = OneClassSVM(kernel='linear')
        print ("Fitting...")
        start = time.time()
        clf.fit(X_res, y_res)
        print ("Fitting time: ", time.time() - start, " seconds")
        print ("Scoring...")
        start = time.time()
        predicted = clf.predict(X_test)
        print ("Scoring time: ", time.time() - start, " seconds")
        del clf
        return (predicted)

    def save_classification_score(self, y_test, predicted):
        print (classification_report(y_test,predicted))
        print (confusion_matrix(y_test,predicted))

if __name__ == "__main__":
    samp = Sampling()
    samp.controller()
