import Class
import ClassificationInfo

from enum import Enum
class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class Learner: 

    def __init__(self, data, className, id, idName):
        self.data = data
        self.size = data.shape[0]
        self.classPlace = className # send in the name of the class column or the index
        self.classesData = {}
        self.id = id
        self.idPlace = idName # send in the name of the id column or the index
        self.train()
    
    def train(self):
        #remove id column if it exists
        if(self.id):
            self.data = self.data.drop(self.idPlace, axis=1)

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
        total = 0
        correct = 0
        #remove id column if it exists
        if(self.id):
            testData = testData.drop(self.idPlace, axis=1)

        #loop through test data and classify
        for index, row in testData.iterrows(): 
            classifications = []
            classPlace = row[self.classPlace]
            highestProb = 0
            className = ""
            #loop through classes and pull highest probability
            for c in self.classesData.keys():
                prob = self.classesData[c].classify(row.drop(self.classPlace))
                if prob > highestProb:
                    highestProb = prob
                    className = c
            #create classification info and store
            classifications.append(self.accuracy(classPlace, className))
            total += 1
            if(classPlace == className):
                correct += 1
        print("Percentage correct: ", correct/total)
        return classifications
    
    def accuracy(self, trueClass, assignedClass):
        classNames = list(self.classesData.keys())
        #decide where classification falls on confusion matrix
        if trueClass == assignedClass:
            if trueClass == classNames[0]:
                accuracy =  Accuracy.TP
            else:
                accuracy = Accuracy.TN
        else:
            if assignedClass == classNames[0]:
                accuracy = Accuracy.FP
            else:
                accuracy = Accuracy.FN
        return ClassificationInfo.ClassificationInfo(trueClass, assignedClass, accuracy)