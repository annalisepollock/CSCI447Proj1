import ClassificationInfo

class AlgorithmAccuracy:
    def __init__(self, stats):
        if not isinstance(stats, ClassificationInfo.ClassificationInfo):
            raise TypeError('stats must be an instance of ClassificationInfo')
        self.stats = stats
        print("True Positives", stats.TP)
        print("True Negatives", stats.TN)
        print("False Positives", stats.FP)
        print("False Negatives", stats.FN)
        #calculate precision, recall, f1
        self.precision = (self.stats.TP)/(self.stats.TP + self.stats.FP)
        self.recall = (self.stats.TP)/(self.stats.TP + self.stats.FN)
        print("Precision", self.precision)
        print("Recall", self.recall)

        self.f1 = 2 * ((self.precision * self.recall)/(self.precision + self.recall))
        self.loss = self.calculateLoss()
    def confusionMatrix(self):
        return {"TP": self.stats.TP, "FP": self.stats.FP, "FN": self.stats.FN, "TN": self.stats.TN}
    
    #calculate loss by summing the number of correct classifications and dividing by the total number of classifications
    def calculateLoss(self):
        total = 0
        sum = 0
        for i in self.stats.getTrueClasses():
            if i[0] == i[1]:
                sum += 1
            else:
                sum += 0
            total += 1
        return sum/total
    
    def getF1(self):
        return self.f1
    def getLoss(self):
        return self.loss
    
    def print(self):
        print("F1: " + str(self.f1))
        print("Loss: " + str(self.loss))
        print("Confusion Matrix: " + str(self.confusionMatrix()))