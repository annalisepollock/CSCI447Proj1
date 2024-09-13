import Class
import ClassificationInfo

from enum import Enum
class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class Learner: 

    def __init__(self, data, className, classification):
        if not isinstance(classification, ClassificationInfo.ClassificationInfo):
            raise TypeError('classification must be an instance of ClassificationInfo')
        self.data = data
        self.size = data.shape[0]
        self.classPlace = className # send in the name of the class column or the index
        self.classesData = {}
        self.classification = classification
        self.train()

    def train(self):
        #loop through data and split into classes
        for index, row in self.data.iterrows():
            if row[self.classPlace] not in self.classesData.keys():
                #create instance of Class
                classify = Class.Class(row[self.classPlace], self.size, row.drop(self.classPlace))
                self.classesData[row[self.classPlace]] = classify
            #add row to data for respective class
            else:
                self.classesData[row[self.classPlace]].add_data(row.drop(self.classPlace))
        #calculate attribute probabilities for each class
        for c in self.classesData.keys():
            self.classesData[c].createAttributes()

    def classify(self, testData, toPrint=False):
        total = 0
        correct = 0
        #loop through test data and classify
        for index, row in testData.iterrows(): 
            if toPrint:
                print("Classifying example:\n", row)
            trueClass = row[self.classPlace]
            highestProb = 0
            predClass = ""
            #loop through classes and pull highest probability
            for c in self.classesData.keys():
                prob = self.classesData[c].classify(row.drop(self.classPlace))
                if prob > highestProb:
                    highestProb = prob
                    predClass = c
            if toPrint:
                print("\tClassified as class:", predClass)
                print("\tTrue class:", trueClass)
                print()
            if trueClass == predClass:
                correct += 1
            total += 1
            #add classification to classification info
            self.classification.addConfusion(self.accuracy(trueClass, predClass))
            self.classification.addTrueClass([trueClass, predClass])
        if toPrint:
            print("TOTAL CLASSIFIED CORRECT:", (correct/total))
            print()
        return correct/total

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
    
    def printInfo(self):
        print()
        print("CLASS NAMES AND VALUES:")
        for className in self.classesData.keys():
            print("NAME:", className)
            print("Total Class Values: ", self.classesData[className].vals)
            print("Class Probability: ", self.classesData[className].getProbability())
            print()
            print("Attribute information:")
            self.classesData[className].printAttributes()
            print()
