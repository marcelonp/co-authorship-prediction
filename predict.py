import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as met
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.features.importances import FeatureImportances

ABSPATH = os.path.dirname(os.path.abspath(__file__))


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == "Connected":
            TP += 1
        if y_hat[i] == "Connected" and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == "Not Connected":
            TN += 1
        if y_hat[i] == "Not Connected" and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


class Predictor():

    def __init__(self):
        self.years = None
        self.df = None
        self.cols = None
        self.setup()

    def setup(self):
        df = pd.read_csv(ABSPATH + "/base.csv")
        drop = []
        for col in df.columns:
            if col.endswith("p"):
                drop.append(col)

        df["Classe"] = ["Connected" if inst ==
                        1 else "Not Connected" for inst in df["Classe"]]

        drop.append("kwds")
        df = df.drop(columns=drop)
        print(df.columns)

        self.years = sorted(set(df["Year"]))
        self.years.remove(2017)
        self.df = df

    def setupyears(self, year):

        df = self.df.copy()
        self.cols = list(df.columns[5:])
        X_train = df[(df["Year"] >= int(year)) & (
            df["Year"] <= 2016)][df.columns[5:]].values
        y_train = df[(df["Year"] >= int(year)) & (
            df["Year"] <= 2016)][df.columns[4]].values
        # print("X_train",X_train.shape)
        # print("y_train",y_train.shape)

        X_test = df[df["Year"] > 2016][df.columns[5:]].values
        y_test = df[df["Year"] > 2016][df.columns[4]].values
        # print("X_test",X_test.shape)
        # print("y_test",y_test.shape)
        # rint(df.columns[5:])

        print(year, " Split:")
        print("TRAIN:", 1 - (X_test.shape[0] /
                             (X_train.shape[0]+X_test.shape[0])))
        print("TEST:", X_test.shape[0]/(X_train.shape[0]+X_test.shape[0]))

        X_train, X_test = self.scale(X_train, X_test)
        return (X_train, y_train, X_test, y_test)

    def scale(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return (X_train_scaled, X_test_scaled)

    def evaluate_multiple(self):
        models = []
        models.append(('DT', DecisionTreeClassifier()))
        models.append(
            ('RF', RandomForestClassifier(n_estimators=100, n_jobs=-1)))
        models.append(('LR', LogisticRegression(solver='lbfgs', n_jobs=-1)))
        models.append(('MLPC', MLPClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM-linear', LinearSVC()))

        results = pd.DataFrame(columns=["Classifier", "Year", "Accuracy",
                                        "Precision", "Recall", "F1", "AUC", "PR-AUC", "TP", "TN", "FP", "FN"])

        for year in self.years:
            for name, model in models:
                X_train, y_train, X_test, y_test = self.setupyears(year)
                print("Fitting ", name, "...", year)
                startfit = time.time()
                model.fit(X_train, y_train)
                fittime = round(time.time()-startfit, 3)
                print("Predicting ", name, "...", year)
                startpred = time.time()
                predicted = model.predict(X_test)
                predtime = round(time.time()-startpred, 3)

                if name != "SVM-linear":
                    predicted_proba = model.predict_proba(X_test)[:, 0]
                else:
                    predicted_proba = model.decision_function(X_test)

                try:
                    viz = FeatureImportances(
                        model, labels=self.cols, absolute=True)
                    viz.fit(X_train, y_train)
                    viz.poof(outpath=ABSPATH+"/other/" +
                             str(name)+str(year)+"import.png")
                    plt.clf()
                except:
                    pass

                print("Scoring ", name, "...", year)
                startsco = time.time()
                print(met.confusion_matrix(y_test, predicted))
                acc = met.accuracy_score(y_test, predicted)
                prec = met.precision_score(
                    y_test, predicted, pos_label="Connected")
                f1 = met.f1_score(y_test, predicted, pos_label="Connected")
                rec = met.recall_score(
                    y_test, predicted, pos_label="Connected")
                auc = met.roc_auc_score(y_test, predicted_proba)
                pr_auc = met.average_precision_score(
                    y_test, predicted_proba, pos_label="Connected")
                tp, fp, tn, fn = perf_measure(y_test, predicted)
                scotime = round(time.time()-startsco, 3)

                dic = {"Classifier": name,
                       "Year": year,
                       "Accuracy": round(acc, 3),
                       "Precision": round(prec, 3),
                       "Recall": round(rec, 3),
                       "F1": round(f1, 3),
                       "AUC": round(auc, 3),
                       "PR-AUC": round(pr_auc, 3),
                       "TP": tp,
                       "TN": tn,
                       "FP": fp,
                       "FN": fn,
                       "fittime": fittime,
                       "predtime": predtime,
                       "scotime": scotime}
                print(dic)

                results = results.append(dic, ignore_index=True)
                results.to_csv(ABSPATH + "/results.csv", index=False)
    # box plot comparison

    def evaluate_prolificos(self):

        df = self.df
        X_train = df[(df["Year"] >= int(2000)) & (
            df["Year"] <= 2016)][df.columns[5:]].values
        y_train = df[(df["Year"] >= int(2000)) & (
            df["Year"] <= 2016)][df.columns[4]].values

        X_test = df[df["Year"] > 2016][df.columns[5:]].values
        y_test = df[df["Year"] > 2016][df.columns[4]].values

        X_train, X_test = self.scale(X_train, X_test)

        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        prec = met.precision_score(y_test, predicted, pos_label="Connected")
        f1 = met.f1_score(y_test, predicted, pos_label="Connected")
        rec = met.recall_score(y_test, predicted, pos_label="Connected")
        print(prec, rec, f1)
        df17 = self.df[self.df["Year"] == 2017]
        df17["Predicted"] = predicted

        results = pd.DataFrame(
            columns=["Author", "Precision", "Recall", "F1", "TP", "TN", "FP", "FN"])
        cod_prolific = [89, 117, 136, 254, 317, 324,
                        399, 184, 419, 469, 485, 519, 533, 552, 561]

        for p in cod_prolific:
            print("author: ", p)
            stest = list(df17[(df17["Author1"] == p) |
                              (df17["Author2"] == p)]["Classe"])
            spred = list(df17[(df17["Author1"] == p) | (
                df17["Author2"] == p)]["Predicted"])
            print(df17[((df17["Author1"] == p) | (df17["Author2"] == p)) & (
                df17["Classe"] == "Connected")])
            prec = met.precision_score(stest, spred, pos_label="Connected")
            f1 = met.f1_score(stest, spred, pos_label="Connected")
            rec = met.recall_score(stest, spred, pos_label="Connected")
            tp, fp, tn, fn = perf_measure(stest, spred)
            dic = {"Author": p,
                   "Precision": round(prec, 3),
                   "Recall": round(rec, 3),
                   "F1": round(f1, 3),
                   "TP": tp,
                   "TN": tn,
                   "FP": fp,
                   "FN": fn,
                   }
            results = results.append(dic, ignore_index=True)
            results.to_csv(ABSPATH + "/results_prolific.csv", index=False)


if __name__ == "__main__":
    #X_train, y_train, X_test, y_test = setup()
    #evaluate(X_train, y_train, X_test, y_test)
    p = Predictor()
    p.evaluate_prolificos()
