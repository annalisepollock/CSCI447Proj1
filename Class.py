import pandas as pd
class Class:
    def __init__(self, name, totalVals, inputData):
        self.name = name
        self.data = pd.DataFrame(inputData)
        self.data = self.data.transpose()
        self.vals = 1
        self.totalVals = totalVals
        self.attributes = {}

    # add a row of data that belongs to class
    def add_data(self, inputData):
        data_df = pd.DataFrame([inputData])
        self.data = pd.concat([self.data, data_df])
        self.vals += 1
    
    def printData(self):
        print(self.data)
    
    def getProbability(self):
        return self.vals/self.totalVals

    def createAttributes(self):
        total = self.data.shape[0] #total number of rows
        #loop through attributes and calculate probabilities
        for columnName, columnData in self.data.items():
            columnInfo = columnData.value_counts() # pull value counts for each attribute possibility
            self.attributes[columnName] = {} #for each attribute create a dictionary with possible values 
            for value, count in columnInfo.items():
                # calculate probability of each attribute value
                self.attributes[columnName][value] = (count + 1) /(len(columnInfo) + total)
    
    #calculate probability that example belongs to class
    def classify(self, example):
        numAttributes = self.data.shape[1] #number of attributes
        total = self.data.shape[0]
        #classify the example
        prob = self.getProbability()
        atributeProb = 1
        #multiply the conditional probability of each attribute value for this class
        for columnName, value in example.items():
            if value in self.attributes[columnName]:
                atributeProb *= self.attributes[columnName][value]

            #if value not present in class multiply with the assumption the count is 0
            else:
                atributeProb *= 1 / (numAttributes + total)
        prob *= atributeProb #multiply 
        return prob
    
    def printAttributes(self):
        for attribute in self.attributes.keys():
            print("Attribute Name", attribute)
            for value in self.attributes[attribute].keys():
                print("\tAttribute Value:", value)
                print("\t\tAttribute Probability",self.attributes[attribute][value] )
            print()