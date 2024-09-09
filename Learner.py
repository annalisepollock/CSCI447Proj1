import Class
import ClassificationInfo

from enum import Enum
class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class Learner: 

    def __init__(self, data, classPlace, id, idPlace):
        self.data = data
        self.size = data.shape[0]
        self.classPlace = classPlace
        self.classesData = {}
        self.id = id
        self.idPlace = idPlace
        self.train()
    
    def train(self):
        #train the model
        if(self.id):
            self.data = self.data.drop(self.idPlace, axis=1)
        classes = []
        for index, row in self.data.iterrows():
            if row[self.classPlace] not in classes:
                classes.append(row[self.classPlace])
                classify = Class.Class(row[self.classPlace], self.size, self.data)
                self.classesData[row[self.classPlace]] = classify
            self.classesData[row[self.classPlace]].add_data(row)
        for c in self.classesData.keys():
            self.classesData[c].createAttributes()

    def classify(self, testData):
        if(self.id):
            testData = testData.drop(self.idPlace, axis=1)
        #for example in testData:
        #    for c in classesData.keys():
        #        classesData[c].classify(example - class column)
        #     answer = max(classesData, key=classesData.get)
        # 
        for index, row in testData.iterrows(): 
            classifications = []
            classPlace = row[self.classPlace]
            print("Classify: ", row.drop(self.classPlace))
            highestProb = 0
            className = ""
            for c in self.classesData.keys():
                prob = self.classesData[c].classify(row.drop(self.classPlace))
                if prob > highestProb:
                    highestProb = prob
                    className = c
            
            classifications.append(self.accuracy(classPlace, className))
        return classifications
    
    def accuracy(self, trueClass, assignedClass):
        classNames = list(self.classesData.keys())
        print(classNames)
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