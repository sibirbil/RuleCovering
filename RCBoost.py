# -*- coding: utf-8 -*-

import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed, cpu_count

class RCBfit:
    
    def __init__(self, CorR = 'C'):
        
        # 'C' for classification, 'R' for regression
        # TODO: Currently it is just classification
        self.CorR = CorR
        self.rules = dict()
        self.initialEstimator = None
        self.nofRMPcalls = 0
        self.c = None
        self.A = None
        
    def initialObject(self):
        return self.initialEstimator
    
    def predict(self, xvals):
        
        # Parallel prediction
        p = cpu_count()
        xsets = np.array_split(xvals, p)
        
        predictions = Parallel(n_jobs=p, prefer="threads")(
            delayed(self.chunkPredict)(x0) for x0 in xsets)
        
        return np.hstack(predictions)
        
    def chunkPredict(self, xvals):
    
        if (self.CorR == 'C'):
            predictions = np.zeros(len(xvals), dtype=int)
        else:
            predictions = np.zeros(len(xvals), dtype=float)
              
        for sindx, x0 in enumerate(xvals):
            totvals = np.zeros(len(self.rules[0][-1]), dtype=float)
            totnum = 0
            trueratios = np.zeros(len(self.rules))
            for rindx, rule in enumerate(self.rules.values()):
                truecount = 0
                # The last value in the list stands for
                # the numbers in each class                
                for clause in rule[:-1]:
                    if (clause[1] == 'l'):
                        if (x0[clause[0]] <= clause[2]):
                            truecount = truecount + 1
                    if (clause[1] == 'r'):
                        if (x0[clause[0]] > clause[2]):
                            truecount = truecount + 1
                # Not the last one (class numbers)
                trueratios[rindx] = truecount/(len(rule)-1)
                if (trueratios[rindx] == 1.0):
                    totvals += rule[-1]
                    totnum += 1

            if (sum(totvals) > 0.0):
                if (self.CorR == 'C'):
                    predictions[sindx] = np.argmax(totvals)
                else:
                    predictions[sindx] = (1.0/totnum)*totvals
            else:
                # DEBUG:
                # This should not happen as we have
                # the initial tree in the column pool
                raise ValueError('ERROR: No clause is satisfied! %f' % x0)
        
        return predictions
    
    def exportRules(self):
        
        for rindx, rule in enumerate(self.rules.values()):
            print('RULE %d:' % rindx)
            # Last compenent stores the numbers in each class            
            for clause in rule[:-1]:
                if (clause[1] == 'l'):
                    print('==> x[%d] <= %.2f' % (clause[0], clause[2]))
                if (clause[1] == 'r'):
                    print('==> x[%d] > %.2f' % (clause[0], clause[2]))

            strarray = '['
            for cn in rule[-1][0:-1]:
                strarray += ('{0:.2f}'.format(cn) + ', ')
            strarray += ('{0:.2f}'.format(rule[-1][-1]) + ']')
                
            print('==> Class numbers: %s' % strarray)     


class RCBoost():
    
    def __init__(self, maxNumOfRMPCalls=100, ccp_alpha=0.0, class_weight=None,
                 criterion='gini', max_depth=10, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 presort='deprecated', random_state=None, splitter='best'):
        
        self.fittedInitEstimator = None
        self.featureNames = None
        self.max_depth = max_depth
            
        self.maxNumOfRMPCalls = maxNumOfRMPCalls
            
        self.estimator = DecisionTreeClassifier(ccp_alpha=ccp_alpha, 
                                                class_weight=class_weight, 
                                                criterion=criterion,
                                                max_depth=max_depth, 
                                                max_features=max_features, 
                                                max_leaf_nodes=max_leaf_nodes,
                                                min_impurity_decrease=min_impurity_decrease,
                                                min_impurity_split=min_impurity_split,
                                                min_samples_leaf=min_samples_leaf,
                                                min_samples_split=min_samples_split,
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                presort=presort,
                                                random_state=random_state,
                                                splitter=splitter)
    
    def getRule(self, fitTree, nodeid):

        left = fitTree.tree_.children_left
        right = fitTree.tree_.children_right
        threshold = fitTree.tree_.threshold
        featurenames = [self.featureNames[i] for i in fitTree.tree_.feature]
    
        def recurse(left, right, child, lineage=None):
            if lineage is None:
                lineage = [child]
            if child in left:
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:
                parent = np.where(right == child)[0].item()
                split = 'r'
    
            # The first in the list shows the feature index
            lineage.append((fitTree.tree_.feature[parent], split,
                            threshold[parent], featurenames[parent]))
    
            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
    
        rule = recurse(left, right, nodeid)
        # Weighted values for each class in leaf comes from tree_
        # These will be later filled with actual numbers
        rule[-1] = fitTree.tree_.value[nodeid][0]
    
        return rule        

    def solveRMP(self, c, A, xinit=np.empty(shape=(0), dtype=float)):
        
        modelopt = gp.Model('RMP')
        modelopt.setParam('OutputFlag', False)
        nofsamples, varsize = np.shape(A)
        rhs = np.ones(nofsamples)
        xopt = modelopt.addMVar(shape=int(varsize),\
                                vtype=GRB.CONTINUOUS, name='xopt')
        
        if (len(xinit) > 0):
            xopt.start = np.zeros(varsize)
            for i in range(len(xinit)):
                    xopt[i].start = xinit[i]
                
        modelopt.setObjective(c.T @ xopt, GRB.MINIMIZE)
        modelopt.addConstr(A @ xopt >= rhs, name='constraints')
        modelopt.optimize()
        # DEBUG:
        # print(modelopt.getAttr(GRB.Attr.ObjVal))
        
        primals = xopt.X
        duals = np.array(modelopt.getAttr(GRB.Attr.Pi))
        
        return primals, duals
    
    def fit(self, X, y):
        
        # RCB currently supports only classification
        # TODO: Add regression
        fittedRCB = RCBfit(CorR = 'C')
        
        # Initial estimator is also stored as an output
        fittedRCB.initialEstimator = copy.deepcopy(self.estimator)

        nOfSamples, nOfFeatures = np.shape(X)
        weights = np.ones(nOfSamples)
        nOfClasses = int(max(y) + 1) # classes start with 0
        
        # Initial tree is created
        fitTree = self.estimator.fit(X, y, sample_weight=weights)
        
        # Currently it is just Gini and Entropy
        # TODO: Add other criteria 
        criterion = self.estimator.get_params()['criterion']        
    
        self.featureNames = ['x[' + str(indx) + ']'
                             for indx in range(nOfFeatures)]

        c = np.empty(shape=(0), dtype=np.float)
        rows = np.empty(shape=(0), dtype=np.int32)
        cols = np.empty(shape=(0), dtype=np.int32)
        ruleno = 0
        # Tells us which sample is in which leaf
        y_rules = fitTree.apply(X)
        for leafno in np.unique(y_rules):
            covers = np.where(y_rules == leafno)[0]
            leafyvals = y[covers]  # y values of the samples in the leaf
            unique, counts = np.unique(leafyvals, return_counts=True)
            probs = counts/np.sum(counts)
            # Currently it is just Gini and Entropy
            if (criterion == 'gini'):            
                cost = 1 + (1 - np.sum(probs**2)) # 1 + Gini
            else:
                cost = 1 + (-np.dot(probs, np.log2(probs))) # 1 + Entropy
            rows = np.hstack((rows, covers))
            cols = np.hstack((cols, np.ones(len(covers), dtype=np.int8)*ruleno))
            c = np.append(c, cost)
            rule = self.getRule(fitTree, leafno)
            fittedRCB.rules[ruleno] = rule
            # Fill the last element in 'rule'
            # with actual numbers in each class
            # not the weighted numbers
            numsinclasses = np.zeros(nOfClasses)
            for indx, i in enumerate(unique):
                numsinclasses[int(i)] = counts[indx]
            fittedRCB.rules[ruleno][-1] = numsinclasses            
            ruleno += 1
        
        data = np.ones(len(rows), dtype=np.int8)
        A = csr_matrix((data, (rows, cols)), dtype=np.int8)
        
        t = 0 # In case no RMP calls is reuired
        for t in range(self.maxNumOfRMPCalls):
                
                # Here we solve an LP with warm-start.
                # TODO: Uing the previous optimal basis
                if (t==0):
                    xinit, duals = self.solveRMP(c, A)
                else:
                    xinit, duals = self.solveRMP(c, A, xinit)
                
                weights += duals
                
                fitTree = self.estimator.fit(X, y, sample_weight=weights)
                
                y_rules = fitTree.apply(X)
                FLAG = True
                for leafno in np.unique(y_rules):
                    covers = np.where(y_rules == leafno)[0]
                    leafyvals = y[covers]  # yvals of the samples in the leaf
                    unique, counts = np.unique(leafyvals, return_counts=True)
                    probs = counts/np.sum(counts)
                    # Currently it is just Gini and Entropy
                    if (criterion == 'gini'):
                        cost = 1 + (1 - np.sum(probs**2)) # 1 + Gini
                    else:
                        cost = 1 + (-np.dot(probs, np.log2(probs))) # 1 + Entropy
                    redcost = cost - np.sum(duals[covers])
                    if (redcost < 0):
                        FLAG = False
                        c = np.append(c, cost)
                        rows = np.hstack((rows, covers))
                        cols = np.hstack((cols, np.ones(len(covers), dtype=np.int8)*ruleno))
                        rule = self.getRule(fitTree, leafno)
                        fittedRCB.rules[ruleno] = rule
                        
                        # Fill the last element in 'rule'
                        # with actual numbers in each class
                        # not the weighted numbers
                        numsinclasses = np.zeros(nOfClasses)
                        for indx, i in enumerate(unique):
                            numsinclasses[int(i)] = counts[indx]
                        fittedRCB.rules[ruleno][-1] = numsinclasses
                        ruleno += 1
                
                # FUTURE RESEARCH: Column pool management with removing columns with
                # 'high' positive reduced cost
                
                data = np.ones(len(rows), dtype=np.int8)
                A = csr_matrix((data, (rows, cols)), dtype=np.int8)
                
                if (FLAG):
                    # No column with negative reduced cost
                    break
        
        # TODO: Clean the redundant rules. This could 
        # help interpretability.

        fittedRCB.nofRMPcalls = t
        fittedRCB.c = c
        fittedRCB.A = A
        
        # FUTURE RESEARCH: One way of using the final solution for assigning weights to
        # the rules. For example:
        # if (t != 0):
        #     for j in range(0, len(xinit)):
        #         fittedRCB.rules[j][-1] *= np.exp(xinit[j])        
            
        return fittedRCB