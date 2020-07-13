# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import RuleCoverDatasets as RCDS

from MIRCO import MIRCO

# Test problems
problems = [RCDS.banknote, RCDS.ILPD, RCDS.ionosphere,
            RCDS.transfusion, RCDS.liver, RCDS.tictactoe,
            RCDS.wdbc, RCDS.mammography, RCDS.diabetes, 
            RCDS.oilspill, RCDS.phoneme, RCDS.seeds, RCDS.wine,
            RCDS.glass, RCDS.ecoli]

fname = 'MIRCO_results.txt'
crit = 'gini'
randomstate = 25

for problem in problems:
    
    pname = problem.__name__.upper()
    print(pname)
    
    df = np.array(problem('datasets/'))
    X = df[:, 0:-1]
    y = df[:, -1]
    
    # Initializing Classifiers
    DTestimator = DecisionTreeClassifier(random_state=randomstate, criterion=crit)
    RFestimator = RandomForestClassifier(random_state=randomstate, criterion=crit)
    # Setting up the parameter grids
    DT_pgrid = {'max_depth': [5, 10, 20]}
    
    RF_pgrid = {'max_depth': [5, 10, 20],
                'n_estimators': [10, 50, 100]}
    
    scores = {'DT': [], 'RF': [], 'MIRCO': []}
    nofrules = {'DT': [], 'RF': [], 'MIRCO': []}
    fracofmissed = []
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=randomstate)
    foldnum = 0    
    for train_index, test_index in skf.split(X, y):
        
        foldnum += 1
        print('Fold number: ', foldnum)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=randomstate)
        for pgrid, est, name in zip((DT_pgrid, RF_pgrid),
                                    (DTestimator, RFestimator),
                                    ('DT', 'RF')):
            gcv = GridSearchCV(estimator=est,
                                param_grid=pgrid,
                                scoring='accuracy',
                                n_jobs=1,
                                cv=inner_cv,
                                verbose=0,
                                refit=True)
            gcv_fit = gcv.fit(X_train, y_train)
            
            # Evaluate with the best estimator
            gcv_pred = gcv_fit.best_estimator_.predict(X_test)
            scores[name].append(accuracy_score(gcv_pred, y_test))
            if (name == 'DT'):
                nofrules[name].append(gcv_fit.best_estimator_.tree_.n_leaves)
            else:  # RF
                # Only the results with the heuristic are reported
                solver = 'heu'
                MIRCO_estimator = MIRCO(gcv_fit.best_estimator_, solver)
                mirco_fit = MIRCO_estimator.fit(X_train, y_train)
                MIRCO_pred = mirco_fit.predict(X_test)
                scores['MIRCO'].append(accuracy_score(MIRCO_pred, y_test))
                nofrules['MIRCO'].append(mirco_fit.numOfRules)
                fracofmissed.append(mirco_fit.numOfMissed/len(y_test))
                # Rules in RF are already returned by MIRCO
                nofrules[name].append(mirco_fit.initNumOfRules)
    
    with open(fname, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('Accuracy Scores:', file=f)
        print(scores, file=f)
        print('Number of Rules:', file=f)
        print(nofrules, file=f)
        print('Fractions of Missed Points by MIRCO:', file=f)
        print(fracofmissed, file=f)
        print('<---\n', file=f)