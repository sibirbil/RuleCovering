# -*- coding: utf-8 -*-

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed, cpu_count

class MIRCOfit:


    def __init__(self, CorR = 'C'):
        
        self.rules = dict()
        self.CorR = CorR # 'C' for classification, 'R' for regression
        self.numOfMissed = 0
        self.missedXvals = []
        self.initNumOfRules = 0
        self.numOfRules = 0


    def predict(self, xvals):
        
        # Parallel prediction
        p = cpu_count()
        xsets = np.array_split(xvals, p)
        
        chunkPreds = Parallel(n_jobs=p, prefer="threads")(
            delayed(self.chunkPredict)(x0) for x0 in xsets)
        
        if (self.CorR == 'C'):
            predictions = np.empty(shape=(0), dtype=int)
        else:
            predictions = np.empty(shape=(0), dtype=float)
        
        for indx in range(len(chunkPreds)):
            predictions = np.append(predictions, chunkPreds[indx]['predictions'])
            for x0 in chunkPreds[indx]['missedXvals']:
                self.missedXvals.append(x0)
            self.numOfMissed += chunkPreds[indx]['numOfMissed']

        # DEBUG:
        # if (self.numOfMissed > 0):
        #     print('Warning...')
        #     print('Total number of missed points:' + str(self.numOfMissed))

        return predictions        


    def chunkPredict(self, xvals):
        
        chunkPreds = dict()
        if (self.CorR == 'C'):
            chunkPreds['predictions'] = np.zeros(len(xvals), dtype=int)
        else:
            chunkPreds['predictions'] = np.zeros(len(xvals), dtype=float)
        chunkPreds['numOfMissed'] = 0
        chunkPreds['missedXvals'] = []
        
        for sindx, x0 in enumerate(xvals):
            totvals = np.zeros(len(self.rules[0][-1]), dtype=float)
            approxvals = np.zeros(len(self.rules[0][-1]), dtype=float)
            totnum, approxnum = 0, 0
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
                else:
                    approxvals += trueratios[rindx]*rule[-1]
                    approxnum += 1
                    
            # TODO: We may return the prediction probabilities
            if (sum(totvals) > 0.0):
                if (self.CorR == 'C'):
                    chunkPreds['predictions'][sindx] = np.argmax(totvals)
                else:
                    chunkPreds['predictions'][sindx] = (1.0/totnum)*totvals
            else:
                if (self.CorR == 'C'):
                    chunkPreds['predictions'][sindx] = np.argmax(approxvals)
                else:
                    chunkPreds['predictions'][sindx] = (1.0/approxnum)*approxvals
                
                chunkPreds['missedXvals'].append(x0)
                chunkPreds['numOfMissed'] += 1
                  
        return chunkPreds


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



            
class MIRCO:


    def __init__(self, rf, solver='heu'):
        
        # rf is a fitted Random Forest!        
        self.rf = rf
        self.solver = solver
        self.estimator = None
        self.featureNames = None


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
        # These are later filled with actual numbers instead of weights
        rule[-1] = fitTree.tree_.value[nodeid][0]
    
        return rule   


    def greedySCP(self, c, A):
        
        # TODO: Can be faster by using heaps
        
        # Mathematical model
        # minimize     c'x
        # subject to   Ax >= 1
        #              x in {0,1}
        # c: n x 1
        # A: m x n
        
        # number of rows and number of columns
        m, n = A.shape
        # set of rows (items)
        M = set(range(m))
        # set of columns (sets)
        N = set(range(n))
        
        R = M
        S = set()
        while (len(R) > 0):
            minratio = np.Inf
            for j in N.difference(S):
                # Sum of covered rows by column j
                denom = np.sum(A[list(R), j])
                if (denom == 0):
                    continue
                ratio = c[j]/denom
                if (ratio < minratio):
                    minratio = ratio
                    jstar = j
            column = A[:, jstar]
            Mjstar = set(np.where(column.toarray() == 1)[0])
            R = R.difference(Mjstar)
            S.add(jstar)
    
        listS = list(S)
        # Sort indices
        sindx = list(np.argsort(c[listS]))
        S = set()
        totrow = np.zeros((m, 1), dtype=np.int32)
        for i in sindx:
            S.add(listS[i])
            column = A[:, listS[i]]
            totrow = totrow + column
            if (np.sum(totrow > 0) >= m):
                break
    
        return S


    def solveSCP(self, c, A, solver):
        
        # Number of rows and number of columns
        m, n = np.shape(A)
        
        S = self.greedySCP(c, A)
        S = np.array(list(S), dtype=np.long)

        # The results in the paper are reported with the greeedy heuristic
        # The following are two options that require solving
        # integer programming (IP) problems:
        #   "app" solves an IP problem over the columns obtained with the greedy heuristic
        #   "opt" sovles an IP orıblem over all columns
        
        if (solver == "app" or solver == "opt"):
            modelopt = gp.Model("SCP")
            rhs = np.ones(m)
            if (solver == "app"):
                # Only the columns in S
                A = A[:, S]
                c = c[S]
                xopt = modelopt.addMVar(shape=int(len(S)),
                                        vtype=GRB.BINARY, name="xopt")
                for i in range(len(S)):
                    xopt[i].start = 1
            else:
                # All columns
                xopt = modelopt.addMVar(shape=int(n),
                                        vtype=GRB.BINARY, name="xopt")
                for i in S:
                    xopt[i].start = 1
    
            modelopt.setObjective(c.T @ xopt, GRB.MINIMIZE)
            modelopt.addConstr(A @ xopt >= rhs, name="constraints")
            modelopt.optimize()
            if (solver == "app"):
                S = S[np.where(xopt.X == 1)[0]]    
            else:
                S = np.where(xopt.X == 1)[0]
                
        return S


    def fit(self, X, y):
        
        if (isinstance(self.rf, RandomForestClassifier)):
            fittedMIRCO = MIRCOfit(CorR = 'C')
        else:
            fittedMIRCO = MIRCOfit(CorR = 'R')
        
        nOfSamples, nOfFeatures = np.shape(X)
        nOfClasses = int(max(y) + 1) # classes start with 0
        
        self.featureNames = ['x[' + str(indx) + ']'
                     for indx in range(nOfFeatures)]

        criterion = self.rf.criterion
        
        # Total number of rules
        nOfRules = 0
        for fitTree in self.rf.estimators_:
            nOfRules += fitTree.get_n_leaves()
        
        # Initial number of rules is stored
        fittedMIRCO.initNumOfRules = nOfRules
        
        # Parallel construction of SCP matrices
        p = cpu_count()
        estsets = np.array_split(self.rf.estimators_, p)
        
        retdicts = Parallel(n_jobs=p, prefer="threads")(
            delayed(self.chunkFit)(X, y, est, criterion, fittedMIRCO.CorR)
            for chunkNo, est in enumerate(estsets))
        
        c = np.empty(shape=(0), dtype=np.float)
        rows = np.empty(shape=(0), dtype=np.int32)
        cols = np.empty(shape=(0), dtype=np.int32)
        colTreeNos = np.empty(shape=(0), dtype=np.int32)
        colLeafNos = np.empty(shape=(0), dtype=np.int32)
        colChunkNos = np.empty(shape=(0), dtype=np.int32)
        colno = 0
        for chunkNo in range(len(estsets)):
            ncols = len(retdicts[chunkNo]['c'])
            c = np.hstack((c, retdicts[chunkNo]['c']))
            rows = np.hstack((rows, retdicts[chunkNo]['rows']))
            colTreeNos = np.hstack((colTreeNos, retdicts[chunkNo]['colTreeNos']))
            colLeafNos = np.hstack((colLeafNos, retdicts[chunkNo]['colLeafNos']))
            tempcols = colno + retdicts[chunkNo]['cols']
            cols = np.hstack((cols, tempcols))
            colChunkNos = np.hstack((colChunkNos, np.ones(ncols,
                                                        dtype=np.int8)*chunkNo))
            colno = cols[-1]+1
        

        data = np.ones(len(rows), dtype=np.int8)
        A = csr_matrix((data, (rows, cols)), dtype=np.int8)
                
        S = self.solveSCP(c, A, self.solver)
        
        for indx, col in enumerate(S):
            chunkno = colChunkNos[col]
            treeno = colTreeNos[col]
            fitTree = estsets[chunkno][treeno]
            leafno = colLeafNos[col]
            rule = self.getRule(fitTree, leafno)
            fittedMIRCO.rules[indx] = rule
            
            # Filling the last element in 'rule'
            # with actual numbers in each class
            # not the weighted numbers - Though,
            # we do not use weights for MIRCO
            y_rules = fitTree.apply(X)
            covers = np.where(y_rules == leafno)
            leafyvals = y[covers]  # yvals of the samples in the leaf
            unique, counts = np.unique(leafyvals, return_counts=True)
            numsinclasses = np.zeros(nOfClasses)
            for ix, i in enumerate(unique):
                numsinclasses[int(i)] = counts[ix]
            fittedMIRCO.rules[indx][-1] = numsinclasses
            
        fittedMIRCO.numOfRules = len(S)
        
        return fittedMIRCO


    def chunkFit(self, X, y, estimators, criterion, CorR):
        
        numRules = 0
        for fitTree in estimators:
            numRules += fitTree.get_n_leaves()
        
        retdict = dict()
        
        retdict['c'] = np.zeros(numRules, dtype=np.float)
        retdict['rows'] = np.empty(shape=(0), dtype=np.int32)
        retdict['cols'] = np.empty(shape=(0), dtype=np.int32)

        retdict['colLeafNos'] = np.zeros(numRules, dtype=np.int32)
        retdict['colTreeNos'] = np.zeros(numRules, dtype=np.int32)
        
        col = 0
        for treeno, fitTree in enumerate(estimators):
            # Tells us which sample is in which leaf
            y_rules = fitTree.apply(X)
            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                retdict['rows'] = np.hstack((retdict['rows'], covers))
                retdict['cols'] = np.hstack((retdict['cols'], 
                                             np.ones(len(covers), dtype=np.int8)*col))                
                leafyvals = np.array(y[covers]) # y values of the samples in the leaf
                if (CorR == 'C'): # classification
                    unique, counts = np.unique(leafyvals, return_counts=True)
                    probs = counts/np.sum(counts)
                    # Currently it is just Gini and Entropy
                    # TODO: Add other criteria
                    if (criterion == 'gini'):
                        retdict['c'][col] = 1 + (1 - np.sum(probs**2)) # 1 + Gini
                    else:
                        retdict['c'][col] = 1 + (-np.dot(probs, np.log2(probs))) # 1 + Entropy
                else: # regression
                    # Currently it is just MSE
                    # TODO: Add other criteria
                    leafyavg = np.average(leafyvals)
                    mse = np.average(np.square(leafyavg - leafyvals))
                    if (criterion == 'mse'):
                        retdict['c'][col] = 1.0 + mse # 1 + MSE
                retdict['colLeafNos'][col] = leafno
                retdict['colTreeNos'][col] = treeno
                col += 1
                
        return retdict