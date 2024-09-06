import pandas as pd
import Learner

def main():
    dataFile = open("/Users/annalise/Downloads/breast-cancer-wisconsin.data", "r")
    data = pd.DataFrame(dataFile)[0].str.split(',', expand=True).replace('\n', '', regex=True)
    data = data.head(20)
    #data = cleanData(data, False)
    learner = Learner.Learner(data, data.shape[0])
    info = learner.test(data)


def accuracyStats():
    return 

def cleanData(dataSet, noise):
    if(noise):
        addNoise(dataSet)
    #clean data
    return dataSet

def addNoise(dataSet):
    #add noise to the data set
    return dataSet

main()
