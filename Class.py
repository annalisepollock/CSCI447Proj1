import pandas as pd
class Class:
    def __init__(self, name, totalVals):
        self.name = name
        self.data = pd.DataFrame()
        self.vals = 0
        self.totalVals = totalVals
        self.attributes = {}

    # add a row of data that belongs to class
    def add_data(self, data):
        self.data = self.data.append(data)
        self.vals += 1
    
    def printData(self):
        print(self.data)
    
    def getProbability(self):
        return self.vals/self.totalVals

    def createAttributes(self):
        numAttributes = self.data.shape[1] #number of attributes
        total = self.data.shape[0] #total number of rows

        #loop through attributes and calculate probabilities
        for columnName, columnData in self.data.iteritems():
            columnInfo = columnData.value_counts() # pull value counts for each attribute possibility
            self.attributes[columnName] = {} #for each attribute create a dictionary with possible values 

            for value, count in columnInfo.iteritems():
                # calculate probability of each attribute value
                self.attributes[columnName][value] = (count + 1) /(numAttributes + total)
    
    #calculate probability that example belongs to class
    def classify(self, example):
        #classify the example
        prob = self.vals / self.totalVals
        atributeProb = 1
        for columnName, value in example.iteritems():
            if value in self.attributes[columnName]:
                atributeProb *= self.attributes[columnName][value]
            else:
                atributeProb *= 1
        prob *= atributeProb
        return prob
    def printAttributes(self):
        print(self.attributes)