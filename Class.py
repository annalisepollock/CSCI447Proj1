import pandas as pd
class Class:
    def __init__(self, name, totalVals, fullData):
        self.name = name
        self.data = pd.DataFrame()
        self.vals = 0
        self.totalVals = totalVals
        self.attributes = {}
        self.fullData = fullData

    def add_data(self, data):
        self.data = self.data.append(data)
        self.vals += 1
    
    def printData(self):
        print(self.data)
    
    def getProbability(self):
        return self.vals/self.totalVals

    def createAttributes(self):
        numAttributes = self.data.shape[1] #number of attributes
        total = self.data.shape[0]
        #dont use unnecessary columns 
        for columnName, columnData in self.data.iteritems():
            attributeNum = self.fullData[columnName].nunique() #number of unique attributes
            columnInfo = columnData.value_counts()
            self.attributes[columnName] = {}
            for value, count in columnInfo.iteritems():
                self.attributes[columnName][value] = (count + 1) /(numAttributes + total)
    
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