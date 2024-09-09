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

    breastCancerClean = cleanData(breastCancerData, breastCancerNoNull, False)
    breastCancerClean['Bare_nuclei'] = breastCancerClean['Bare_nuclei'].astype(int)

    # END BREAST CANCER DATASET CLEANING

    #learner = Learner.Learner(data, 10)
    print("Done")

def cleanData(dataOriginal, dataSet, noise):
    dataVariables = pd.DataFrame(dataOriginal.variables)
    print(dataVariables.to_string())
    #dataMetadata = dataSet.metadata

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
        # randomly split data into 10 parts
        # rotate each part to be used as testing 1x
        # call learn, classify, etc.

main()
