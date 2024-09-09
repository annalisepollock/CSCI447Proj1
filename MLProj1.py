import pandas as pd
import Learner
from io import StringIO

def main():
    
    dataFile = open("/Users/annalise/Downloads/breast-cancer-wisconsin.data", "r")
    #grab class location
    data = pd.DataFrame(dataFile)[0].str.split(',', expand=True).replace('\n', '', regex=True)
    train, test = testSet(data)
    learner = Learner.Learner(train, 10, True, 0)
    learner.classify(test)


def accuracyStats():
    return 

def cleanData(dataSet, noise, classPlace):
    
    if(noise):
        addNoise(dataSet)
    #clean data
    return dataSet.drop(classPlace, axis=1)

def addNoise(dataSet):
    #add noise to the data set
    return dataSet

def testSet(dataSet):
    toDrop = dataSet.sample(frac=0.15).index
    test = dataSet.loc[toDrop]
    train = dataSet.drop(toDrop)
    return train, test

main()
