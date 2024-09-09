class ClassificationInfo:
    def __init__(self, trueClass, AssignedClass, Accuracy):
        self.trueClass = trueClass
        self.AssignedClass = AssignedClass
        self.Accuracy = Accuracy

    def getAccuracy(self):
        return self.Accuracy
        