from enum import Enum
class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

class ClassificationInfo:
    def __init__(self, trueClass, AssignedClass, Accuracy):
        self.trueClass = trueClass
        self.AssignedClass = AssignedClass
        self.Accuracy = Accuracy

    def getAccuracy(self):
        return self.Accuracy
        