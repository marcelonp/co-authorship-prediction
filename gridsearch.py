
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
from yellowbrick.classifier import PrecisionRecallCurve

ABSPATH = os.path.dirname(os.path.abspath(__file__))

##############################################
# SCRIPT PARA A EXECUÇÂO DA BUSCA A GRADE E PLOTAGEM DA CURVA DE PRECISAO RECALL
#############################################

# Load the classification data set
data = pd.read_csv(ABSPATH + "/base.csv")

# Specify the features of interest and the target
target = "Classe"

notcols = ["kwds"]
for col in data.columns:
    if col.endswith("p"):
        notcols.append(col)


data = data[data["Year"] >= 2015]
for i, date in enumerate(data["Year"]):
    if date == 2017:
        endtrain = i
        break

#cvtrain = list(range(0,endtrain-1))
#cvtest = list(range(endtrain,len(data)))
#cv = [[cvtrain,cvtest]]

notcols = notcols + list(data.columns[:5])
print(notcols)
features = [col for col in data.columns if col not in notcols]
print(features)

params = {'bootstrap': [True, False],
          'max_depth': [10, 50, 100, None],
          'max_features': ['auto', 'sqrt'],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [100, 250, 500]
          }

features = ["Katz", "RPR", "SR", "CM", "MA"]
feats2 = ['AA', 'CN', 'JC', 'RA', 'Katz', 'RPR',
          'SR', 'CM', 'DC', 'MA', 'KC', 'EC', 'IC']

X_train = data[data["Year"] != 2017][features]
X_train2 = data[data["Year"] != 2017][feats2]
y_train = data[data["Year"] != 2017]["Classe"]

X_test = data[data["Year"] == 2017][features]
X_test2 = data[data["Year"] == 2017][feats2]
y_test = data[data["Year"] == 2017]["Classe"]


print("griding...")
#clf = RandomForestClassifier()
#grid = GridSearchCV(clf,param_grid=params,cv=cv,scoring="f1", n_jobs=-1, verbose=50)
# grid.fit(X.values,y.values)
# print(sorted(grid.cv_results_))
clf1 = RandomForestClassifier(n_estimators=100, max_depth=10,
                              min_samples_split=10, min_samples_leaf=1, bootstrap=False, n_jobs=-1)
clf2 = RandomForestClassifier(n_jobs=-1)


# Load the dataset and split into train/test splits


# Create the visualizer, fit, score, and poof it
viz = PrecisionRecallCurve(clf1)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.poof()

viz = PrecisionRecallCurve(clf2)
viz.fit(X_train2, y_train)
viz.score(X_test2, y_test)
viz.poof()
