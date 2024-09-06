import pandas as pd
class Class:
    def __init__(self, name, totalVals):
        self.name = name
        self.data = pd.DataFrame()
        self.vals = 0
        self.totalVals = totalVals
        self.attributes = {}

    def add_data(self, data):
        self.data = self.data.append(data)
        self.vals += 1
    
    def printData(self):
        print(self.data)
    
    def getProbability(self):
        return self.vals/self.totalVals

    def createAttributes(self):
        numAttributes = self.data.shape[1]
        #dont use unnecessary columns 
        for columnName, columnData in self.data.iteritems():
            columnInfo = columnData.value_counts()
            total = columnData.count()
            self.attributes[columnName] = {}
            for value, count in columnInfo.iteritems():
                self.attributes[columnName][value] = count + 1 /numAttributes + total #fix
    
    def classify(example):
        #classify the example
        return example