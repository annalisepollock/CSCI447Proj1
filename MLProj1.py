import math
import random
import warnings
from pandas.core.common import SettingWithCopyWarning
import pandas as pd
import numpy as np
import Learner
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def main():
    # FETCH DATASETS
    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerColumnTypes = dict(zip(breastCancerData.variables['name'], breastCancerData.variables['type']))

    glassData = fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassColumnTypes = dict(zip(glassData.variables['name'], glassData.variables['type']))

    irisData = fetch_ucirepo(id=53)
    irisDataFrame = pd.DataFrame(irisData.data.original)
    irisColumnTypes = dict(zip(irisData.variables['name'], irisData.variables['type']))
    # END FETCH DATASETS

    # BREAST CANCER DATASET CLEANING
    # curated data cleaning: breast cancer data
    breastCancerNoId = breastCancerDataFrame.drop(columns=['Sample_code_number']) # remove ID column

    breastCancerClean = cleanData(breastCancerData, breastCancerNoId, False)
    breastCancerClean['Bare_nuclei'] = breastCancerClean['Bare_nuclei'].astype(int)
    # END BREAST CANCER DATASET CLEANING

    # GLASS DATASET CLEANING/BINNING
    glassDataFrame = glassDataFrame.drop(columns=['Id_number'])

    # binning, if continuous/categorical
    for columnName in glassDataFrame.columns:
        # TO-DO: ignore Class column
        # if continuous, make categorical (5 categories total)
        if glassColumnTypes[columnName] == 'Continuous':
            # split dataset into 5 equal-width bins
            binVals = np.linspace(glassDataFrame[columnName].min(), glassDataFrame[columnName].max(), 6)
            binLabels = ['A', 'B', 'C', 'D', 'E']

            # assign column vals to a bin
            glassDataFrame[columnName] = pd.cut(glassDataFrame[columnName], bins = binVals, labels = binLabels, include_lowest = True)
            # set 'type' to Categorical
            glassColumnTypes[columnName] = 'Categorical'

        if glassColumnTypes[columnName] == 'Categorical':
            # one-hot encoding
            encoder = OneHotEncoder()

            # encode all columns except for the class column
            glassColumnsToEncode = glassDataFrame.iloc[:, :-1]

            glassDataEncoded = pd.DataFrame(encoder.fit_transform(glassColumnsToEncode).toarray())
            glassClean = pd.concat([glassDataEncoded, glassDataFrame.iloc[:, -1]], axis=1)
    # END GLASS DATASET CLEANING/BINNING

    # CROSS-VALIDATION, TRAINING + TESTING
    breastCancerFolds = crossValidation(breastCancerClean)
    glassFolds = crossValidation(glassClean)

    # f1 = []
    # 0-1 loss = []
    # for each fold:
    # train = data.drop(fold)
    # learner = Leaner.Learner(train, 'class')
    # classifications = learner.classify(fold)
    # stats = AlgorithmAccuracy.AlgorithmAccuracy(classifications)
    # stats.print()
    # f1.append(stats.getF1())
    # 0-1 loss.append(stats.getLoss())

def cleanData(dataOriginal, dataSet, noise, classColumnName):
    dataVariables = pd.DataFrame(dataOriginal.variables)

    if(noise):
        addNoise(dataSet, classColumnName)

    # Remove any rows where all values are null
    dataRemovedNullRows = dataSet.dropna(how = 'all')

    # Columns must have 70% of their values for rows to remain in dataset
    dataRemovedNullCols = dataRemovedNullRows.dropna(axis=1, thresh = math.floor(0.70*dataSet.shape[0]))

    # Iterate through columns; if numerical, fillna with mean
    # if categorical/binary, use forward fill/backfill
    # round values to nearest int so that na's can be filled with this value regardless if continuous or discrete
    dataSetNoNull = dataRemovedNullCols.fillna(round(dataRemovedNullCols.mean()))

    #print(dataSetNoNull.to_string())
    # Continuous attributes - discretize

    return dataSetNoNull

def addNoise(dataSet, classColumnName):
    #calculate 10% of columns
    numCols = int(0.1 * len(dataSet.columns))
    #dont mix class column
    columns = dataSet.columns.drop(classColumnName).to_list()
    # Randomly select 10% of the columns
    selectedColumns = random.sample(columns, numCols)
    #grab data from those columns
    for col in selectedColumns:
        newCol = np.random.permutation(dataSet[col].values)
        dataSet[col] = newCol

    return dataSet

def crossValidation(cleanDataset):
    # array to hold 10 randomly selected groups from the dataset
    dataChunks = [None] * 10
    numRows = math.floor(cleanDataset.shape[0]/10)
    tempDataset = cleanDataset

    for i in range(9):
        # randomly select 1/10 of the dataset, put it in the array
        chunk = tempDataset.sample(n=numRows)
        #print("chunk sample size for " + str(i) + ": " + str(chunk.shape[0]))
        dataChunks[i] = chunk

        # rest of dataset without selected chunk
        tempDataset = tempDataset.drop(chunk.index)

    #print("size of remaining data: " + str(tempDataset.shape[0]))
    # the last chunk might be slightly different size if dataset size is not divisible by 10
    dataChunks[9] = tempDataset

    # rotate each part to be used as testing 1x
    # call learn, classify, etc. on each version of the train/test data
    return dataChunks
main()
