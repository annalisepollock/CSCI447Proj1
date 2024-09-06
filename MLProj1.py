import pandas as pd
import Learner

def main():
    dataFile = open("/Users/annalise/Downloads/breast-cancer-wisconsin.data", "r")
    data = pd.DataFrame(dataFile)[0].str.split(',', expand=True).replace('\n', '', regex=True)
    data = data.head(20)
    print(data)
    #data = cleanData(data, False)
    learner = Learner.Learner(data, 10)
    print("Done")


def cleanData(dataSet, noise):
    if(noise):
        addNoise(dataSet)
    #clean data
    return dataSet

def addNoise(dataSet):
    #add noise to the data set
    return dataSet

main()
