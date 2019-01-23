# -*- coding: utf-8 -*-
"""
@author: gaurav
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('winequality-red.csv')
def classify(x):
    if x<=4:
        x = 0
    elif x>4 and x<7:
        x= 1
    else:
        x= 2
    return x

dataset["quality"] = dataset["quality"].apply(classify)

X = dataset.iloc[:, [1,4,8,9,10]].values
#X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size =0.25, random_state = 42 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# SVM
from sklearn.svm import SVC
classifier = SVC(C=2, kernel='rbf', gamma= 0.5, random_state=42 )

#Random forest
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=15,random_state=42)

# Train classifier
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train, y=Y_train,cv=10)
accuracies.mean()

#Applying grid search to find best params
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C' : [0.5, 1,2,4,7], 'kernel' : ['rbf'], 'gamma': [0.5,1,2,4,7]}]
#grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,
#                           scoring='accuracy',
#                           cv=10 )
#grid_search = grid_search.fit(X_train,Y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

# backward elimination of features
#import statsmodels.formula.api as sm
#X = np.append(arr=np.ones((1599,1)).astype(int), values=X, axis=1 )
#X_opt = X[:, [0,2,5,9,10,11]]
#regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
#regressor_OLS.summary()