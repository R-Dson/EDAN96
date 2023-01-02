import numpy as np
from collections import Counter
from ordered_set import OrderedSet
from graphviz import Digraph
from sklearn import tree, metrics, datasets
from ordered_set import OrderedSet


class ID3RegressionTreePredictor :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2, maxDepth = 100, stopMSE = 0.0) :

        self.__nodeCounter = -1

        self.__dot = Digraph()

        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit
        self.__maxDepth = maxDepth
        self.__stopMSE = stopMSE

        self.__numOfAttributes = 0
        self.__attributes = None
        self.__target = None
        self.__data = None

        self.__tree = None

    def newID3Node(self):
        self.__nodeCounter += 1
        return {'id': self.__nodeCounter, 'splitValue': None, 'nextSplitAttribute': None, 'mse': None, 'samples': None,
                         'avgValue': None, 'nodes': None}


    def addNodeToGraph(self, node, parentid):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != None):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        #print(nodeString)

        return


    def makeDotData(self) :
        return self.__dot

    # stubb that can be extended to a full blown MSE calculation.
    def calcMSE(self, dataIDXs) :
        if len(dataIDXs) == 0:
            return 0.0, 0.0
        mse = 0.0
        avg = 0.0
        #print(list(dataIDXs))
        # avg = np.mean([self.__target[x] for x in dataIDXs])
        for i in dataIDXs:
            avg += self.__target[i]
        avg /= len(dataIDXs)

        for i in dataIDXs:
            mse += (self.__target[i] - avg)**2
        #_targets = self.__target[list(dataIDXs)]
        #avg = np.mean(_targets)
        # mse=0
        # for i in dataIDXs:
        #     mse += (self.__target[i] - avg)**2

        #print("körs")
        
        #mse = sum( (_targets-avg)**2  )
        
        # print("MSE: ", mse, "AVG", avg , "Targets shape : " , _targets.shape)
        return mse, avg

        return mse, avg

    # find the best split attribute out of the still possible ones ('attributes')
    # over a subset of self.__data specified through a list of indices (dataIDXs)
    def findSplitAttrOld(self, attributes, dataIDXs) :

        minMSE = float("inf")

        splitAttr = ''
        splitMSEs = {}
        splitDataIDXs = {}
        splitAverages = {}
        splitMSEsTemp = {}
        splitDataIDXsTemp = {}
        splitAveragesTemp = {}
        """
        attrl = list(self.__attributes.keys())

        for att in attributes:
            omse = 0
            attindex = attrl.index(att)

            for subatt in self.__attributes[att]:

                indxs = []
                for i in dataIDXs:
                    data = self.__data[i]
                    if subatt == data[attindex]:
                        indxs.append(i)
                #print(indxs)
                
                mset, avgt = self.calcMSE(indxs)
                omse += mset

                splitAveragesTemp[subatt] = avgt
                splitMSEsTemp[subatt] = mset
                splitDataIDXsTemp[subatt] = indxs
            if minMSE > omse:
                minMSE = omse
                splitAttr = att
                splitAverages = splitAveragesTemp
                splitDataIDXs = splitDataIDXs
                splitMSEs = splitMSEsTemp
            splitAveragesTemp = {}
            splitDataIDXsTemp = {}
            splitMSEsTemp = {}
        """
        attgroup = {}
        for att in attributes:
            attid = self.__attributesidx[att]
            groups = {}
            for subatt in self.__attributes[att]:
                groups[subatt] = []
            for i in dataIDXs:
                data = self.__data[i]
                #print(data)
                for subatt in self.__attributes[att]:
                    if data[attid] == subatt:
                        groups[subatt].append(i)
            attgroup[att] = groups

        mses = {}
        for g in attgroup:
            msetotal = 0
            for subg in attgroup[g]:
                #if len(attgroup[g][subg]) > 0:
                    mse, avg = self.calcMSE(attgroup[g][subg])
                    splitMSEs[subg] = mse
                    splitAverages[subg] = avg
                    msetotal += mse
            mses[g] = msetotal

        aslist = list(mses)
        ind = np.argmin(aslist)
        splitAttr = aslist[ind]
        minMSE = mses[splitAttr]
        for attg in attgroup:
            for att in attgroup[attg]:
                #print(attgroup[attg][att])
                splitDataIDXs[att] = attgroup[attg][att]
        #print(splitDataIDXs)
        # ****************************************************************************************
        # Provide your code here (and in potentially needed help methods, like self.calcMSE
        #
        # You find the data in self.__data and target values in self.__target
        # The data set for which you should find the best split attribute by
        # calculating the overall MSE for the respective subsets is specified
        # through the parameter 'dataIDXs', i.e. self.__data and self.__target
        # will never need to be altered themselves, and no copies are needed
        # either!
        #
        # Return:
        # - minMSE: the minimal MSE resulting from your calculations
        # - splitAttr: the attribute that, if used as split attribute, gives the minMSE
        # - splitMSEs: a dictionary (keys: attribute values, values: MSEs) with the MSEs
        #              in each subset resulting from the split
        # - splitAveragesFinal: a dictionary (keys: attribute values, values: average values)
        #                       with the average value (prediction) of each subset
        # - splitDataIDXsFinal: a dictionary (keys: attribute values, values: subset data indices)
        #                       with the list of indices for each subset
        #*****************************************************************************************

        return minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs

    # find the best split attribute out of the still possible ones ('attributes')
    # over a subset of self.__data specified through a list of indices (dataIDXs)
    def findSplitAttr(self, attributes, dataIDXs) :

        minMSE = float("inf")
        
        splitAttr = ''
        splitMSEs = {}
        splitDataIDXs = {}
        splitAverages = {}
        
        splitMSETemp = {}
        splitDataIDXsTemp = {}
        SplitAveragesTemp = {}
        
        attList =[]

        for att in self.__attributes:
            attList.append(att)
        
        for att in attributes:
            oMSE = 0
            attIndx = attList.index(att)
            
            for subatt in self.__attributes[att]:
                idxs = []
                for i in dataIDXs:
                    if self.__data[i][attIndx] == subatt:
                        idxs.append(i)
                
                my_mse, my_avg = self.calcMSE(idxs)
                oMSE += my_mse
                splitMSETemp[subatt] = my_mse
                splitDataIDXsTemp[subatt] = idxs
                SplitAveragesTemp[subatt] = my_avg

            if oMSE < minMSE:
                minMSE = oMSE
                splitAttr = att
                splitMSEs = splitMSETemp
                splitDataIDXs = splitDataIDXsTemp
                splitAverages = SplitAveragesTemp
                
            
            splitMSETemp = {}
            splitDataIDXsTemp = {}
            SplitAveragesTemp = {}
        
        # ****************************************************************************************
        # Provide your code here (and in potentially needed help methods, like self.calcMSE
        #
        # You find the data in self.__data and target values in self.__target
        # The data set for which you should find the best split attribute by
        # calculating the overall MSE for the respective subsets is specified
        # through the parameter 'dataIDXs', i.e. self.__data and self.__target
        # will never need to be altered themselves, and no copies are needed
        # either!
        #
        # Return:
        # - minMSE: the minimal MSE resulting from your calculations
        # - splitAttr: the attribute that, if used as split attribute, gives the minMSE
        # - splitMSEs: a dictionary (keys: attribute values, values: MSEs) with the MSEs
        #              in each subset resulting from the split
        # - splitAveragesFinal: a dictionary (keys: attribute values, values: average values)
        #                       with the average value (prediction) of each subset
        # - splitDataIDXsFinal: a dictionary (keys: attribute values, values: subset data indices)
        #                       with the list of indices for each subset
        #*****************************************************************************************

        return minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs

    # the starting point for fitting the tree
    # you should not need to change anything in here
    def fit(self, data, target, attributes):
        # data = x, target = y
        self.__numOfAttributes = len(attributes)
        self.__attributes = attributes
        self.__data = data
        self.__target = target
        self.__attributesidx = {}
        self.__idxattributes = {}
        i = 0

        for att in attributes:
            self.__attributesidx[att] = i
            j = 0

            attid = {}
            idatt = {}
            for a in attributes[att]:
                attid[a] = j
                idatt[j] = a
                j += 1

            self.__idxattributes[i] = [attid, idatt]
            i += 1


        dataIDXs = {j for j in range(len(data))}

        mse, avg = self.calcMSE(dataIDXs)

        attributesToTest = list(self.__attributes.keys())

        self.__tree = self.fit_rek( 0, None, '-', attributesToTest, mse, avg, dataIDXs)

        return self.__tree


    # the recursive tree fitting method
    # you should not need to change anything in here
    def fit_rek(self, depth, parentID, splitVal, attributesToTest, mse, avg, dataIDXs) :

        root = self.newID3Node()

        root.update({'splitValue':splitVal, 'mse': mse, 'samples': len(dataIDXs)})
        currentDepth = depth

        if (currentDepth == self.__maxDepth or mse <= self.__stopMSE or len(attributesToTest) == 0 or len(dataIDXs) < self.__minSamplesSplit):
            root.update({'avgValue':avg})
            self.addNodeToGraph(root, parentID)
            return root

        minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs = self.findSplitAttr(attributesToTest, dataIDXs)


        root.update({'nextSplitAttribute': splitAttr, 'nodes': {}})
        self.addNodeToGraph(root, parentID)

        attributesToTestCopy = OrderedSet(attributesToTest)
        attributesToTestCopy.discard(splitAttr)
        #print(splitAttr)
        #print(self.__attributes[splitAttr])

        for val in self.__attributes[splitAttr] :
            #print("testing " + str(splitAttr) + " = " + str(val))
            if( len(splitDataIDXs[val]) < self.__minSamplesLeaf) :
                root['nodes'][val] = self.newID3Node()
                root['nodes'][val].update({'splitValue':val, 'samples': len(splitDataIDXs[val]), 'avgValue': splitAverages[val]})
                self.addNodeToGraph(root['nodes'][val], root['id'])
                #print("leaf, not enough samples, setting node-value = " + str(splitAverages[val]))

            else :
                root['nodes'][val] = self.fit_rek( currentDepth+1, root['id'], val, attributesToTestCopy, splitMSEs[val], splitAverages[val], splitDataIDXs[val])

        return root

    # Doing a prediction for a data set 'data' (starting method for the recursive tree traversal)
    def predict(self, data) :
        predicted = list()

        for i in range(len(data)) :
            predicted.append(self.predict_rek(data[i], self.__tree))

        return predicted

    # Recursively traverse the tree to find the value for the sample 'sample'
    def predict_rek(self, sample, node) :

        if(node['avgValue'] != None) :
            return node['avgValue']

        attr = node['nextSplitAttribute']
        dataIDX = list(self.__attributes.keys()).index(attr)
        val = sample[dataIDX]
        next = node['nodes'][val]

        return self.predict_rek( sample, next)

    def score(self, data, target) :
        y_pred = self.predict(data)
        y_true = target
        y_true_mean = np.mean(y_true)
        score = 1.0
        u = np.sum([((y_true[i] - y_pred[i])**2) for i in range(len(y_true))])
        v = np.sum([((y_true[i] - y_true_mean)**2) for i in range(len(y_true))])
        score = 1 - u/v
        # ************************************************
        # Implement your score method here
        # ************************************************

        return score

