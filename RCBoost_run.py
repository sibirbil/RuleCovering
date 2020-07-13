# # -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import RuleCoverDatasets as RCDS

from RCBoost import RCBoost

# Test problems
problems = [RCDS.banknote, RCDS.ILPD, RCDS.ionosphere,
            RCDS.transfusion, RCDS.liver, RCDS.tictactoe,
            RCDS.wdbc, RCDS.mammography, RCDS.diabetes, 
            RCDS.oilspill, RCDS.phoneme, RCDS.seeds, RCDS.wine,
            RCDS.glass, RCDS.ecoli]

fname = 'RCBoost_results.txt'
crit = 'gini'
randomstate = 25

for problem in problems:

    pname = problem.__name__.upper()
    print(pname)

    df = np.array(problem('datasets/'))
    X = df[:, 0:-1]
    y = df[:, -1]

    # Initializing Classifiers

    RFestimator = RandomForestClassifier(random_state=randomstate, criterion=crit)
    ADAestimator = AdaBoostClassifier(random_state=randomstate)
    GBestimator = GradientBoostingClassifier(random_state=randomstate)

    # Setting up the parameter grids
    RF_pgrid = {'max_depth': [5, 10, 20],
                'n_estimators': [10, 50, 100]}

    ADA_pgrid = {'base_estimator': [DecisionTreeClassifier(max_depth=5),
                                    DecisionTreeClassifier(max_depth=10),
                                    DecisionTreeClassifier(max_depth=20)],
                  'n_estimators': [10, 50, 100]}

    GB_pgrid = {'max_depth': [5, 10, 20],
                'n_estimators': [10, 50, 100]}

    RCB_grid = {'max_depth': [5, 10, 20],
                'maxNumOfRMPCalls': [5, 10, 50, 100, 200]}

    scores = {'RF': [], 'ADA': [], 'GB': [], 'RCB': [], 'initDT': []}
    nofRMPcalls = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=randomstate)
    foldnum = 0
    for train_index, test_index in skf.split(X, y):

        foldnum += 1
        print('Fold number: ', foldnum)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=randomstate)

        # RCB parameter selection with CV
        bestscore = 0
        for md in RCB_grid['max_depth']:
            for rmpc in RCB_grid['maxNumOfRMPCalls']:
                RCBestimator = RCBoost(max_depth=md,
                                        maxNumOfRMPCalls=rmpc,
                                        criterion=crit)
                avgscore = 0
                for etrain_index, etest_index in inner_cv.split(X_train, y_train):
                    eX_train, eX_test = X_train[etrain_index], X_train[etest_index]
                    ey_train, ey_test = y_train[etrain_index], y_train[etest_index]
                    rcb = RCBestimator.fit(eX_train, ey_train)
                    RCB_pred = rcb.predict(eX_test)
                    acsc = accuracy_score(ey_test, RCB_pred)
                    avgscore += acsc
    
                avgscore /= inner_cv.n_splits
                if (avgscore > bestscore):
                    bestscore = avgscore
                    bestmd = md
                    bestrmpc = rmpc

        # RCB fit
        RCBestimator = RCBoost(max_depth=bestmd,
                                maxNumOfRMPCalls=bestrmpc,
                                criterion=crit)
        rcb = RCBestimator.fit(X_train, y_train)
        RCB_pred = rcb.predict(X_test)
        scores['RCB'].append(accuracy_score(RCB_pred, y_test))
        nofRMPcalls.append(rcb.nofRMPcalls)

        # initDT fit
        dt = rcb.initialEstimator.fit(X_train, y_train)
        DT_pred = dt.predict(X_test)
        scores['initDT'].append(accuracy_score(DT_pred, y_test))

        # Others
        for pgrid, est, name in zip((RF_pgrid, ADA_pgrid, GB_pgrid),
                                    (RFestimator, ADAestimator, GBestimator),
                                    ('RF', 'ADA', 'GB')):
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

    with open(fname, 'a') as f:
        print('--->', file=f)
        print(pname, file=f)
        print('Accuracy Scores:', file=f)
        print(scores, file=f)
        print('Number of RMP calls:', file=f)
        print(nofRMPcalls, file=f)
        print('<---\n', file=f)