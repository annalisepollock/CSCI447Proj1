import math
import pandas as pd
import Learner
from ucimlrepo import fetch_ucirepo

def main():
    # fetch datasets
    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)

    # BREAST CANCER DATASET CLEANING
    # curated data cleaning: breast cancer data
    breastCancerNoId = breastCancerDataFrame.drop(columns=['Sample_code_number']) # remove ID column

    breastCancerClean = cleanData(breastCancerData, breastCancerNoId, False)
    breastCancerClean['Bare_nuclei'] = breastCancerClean['Bare_nuclei'].astype(int)
    # END BREAST CANCER DATASET CLEANING

    breastCancerFolds = crossValidation(breastCancerClean)

    #learner = Learner.Learner(data, 10)
    print("Done")

def cleanData(dataOriginal, dataSet, noise):
    dataVariables = pd.DataFrame(dataOriginal.variables)

    if(noise):
        addNoise(dataSet)

    # Remove any rows where all values are null
    dataRemovedNullRows = dataSet.dropna(how = 'all')

    # Columns must have 90% of their values for rows to remain in dataset
    dataRemovedNullCols = dataRemovedNullRows.dropna(axis=1, thresh = math.floor(0.90*dataSet.shape[0]))

    # Iterate through columns; if numerical, fillna with mean
    # if categorical/binary, use forward fill/backfill
    # round values to nearest int so that na's can be filled with this value regardless if continuous or discrete
    dataSetNoNull = dataRemovedNullCols.fillna(round(dataRemovedNullCols.mean()))

    #print(dataSetNoNull.to_string())
    # Continuous attributes - discretize


    return dataSetNoNull

def addNoise(dataSet):
    # add noise to the data set
    return dataSet

def crossValidation(cleanDataset):
    # array to hold 10 randomly selected groups from the dataset
    dataChunks = [None] * 10
    numRows = math.floor(cleanDataset.shape[0]/10)
    tempDataset = cleanDataset

    for i in range(9):
        # randomly select 1/10 of the dataset, put it in the array
        chunk = tempDataset.sample(n=numRows)
        print("chunk sample size for " + str(i) + ": " + str(chunk.shape[0]))
        dataChunks[i] = chunk

        # rest of dataset without selected chunk
        tempDataset = tempDataset.drop(chunk.index)

    print("size of remaining data: " + str(tempDataset.shape[0]))
    # the last chunk might be slightly different size if dataset size is not divisible by 10
    dataChunks[9] = tempDataset

    # rotate each part to be used as testing 1x
    # call learn, classify, etc. on each version of the train/test data
    return dataChunks

main()
