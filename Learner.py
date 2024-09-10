import Class
import ClassificationInfo

from enum import Enum
class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class Learner: 

    def __init__(self, data, className):
        self.data = data
        self.size = data.shape[0]
        self.classPlace = className # send in the name of the class column or the index
        self.classesData = {}
        self.train()
    
    def train(self):
        #loop through data and split into classes
        for index, row in self.data.iterrows():
            if row[self.classPlace] not in self.classesData.keys():
                #create instance of Class
                classify = Class.Class(row[self.classPlace], self.size)
                self.classesData[row[self.classPlace]] = classify
            #add row to data for respective class
            self.classesData[row[self.classPlace]].add_data(row)
        #calculate attribute probabilities for each class
        for c in self.classesData.keys():
            self.classesData[c].createAttributes()

    def classify(self, testData):
        classification = ClassificationInfo.ClassificationInfo()

        #loop through test data and classify
        for index, row in testData.iterrows(): 
            trueClass = row[self.classPlace]
            highestProb = 0
            predClass = ""
            #loop through classes and pull highest probability
            for c in self.classesData.keys():
                prob = self.classesData[c].classify(row.drop(self.classPlace))
                if prob > highestProb:
                    highestProb = prob
                    predClass = c

            #add classification to classification info
            classification.addConfusion(self.accuracy(trueClass, predClass))
            classification.addTrueClass([trueClass, predClass])

        return classification
    
    def accuracy(self, trueClass, assignedClass):
        classNames = list(self.classesData.keys())
        #decide where classification falls on confusion matrix
        if trueClass == assignedClass:
            if trueClass == classNames[0]:
                return Accuracy.TP
            else:
                return Accuracy.TN
        else:
            if assignedClass == classNames[0]:
                return Accuracy.FP
            else:
                return Accuracy.FN