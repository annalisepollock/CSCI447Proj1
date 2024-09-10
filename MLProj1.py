import math
import random
import warnings
from pandas.core.common import SettingWithCopyWarning
import pandas as pd
import numpy as np
import Learner
import AlgorithmAccuracy
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

def main():
    # FETCH DATASETS
    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)

    glassData = fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)

    irisData = fetch_ucirepo(id=53)
    irisDataFrame = pd.DataFrame(irisData.data.original)

    soybeanData = fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)

    votingData = fetch_ucirepo(id=105)
    votingDataFrame = pd.DataFrame(votingData.data.original)
    # END FETCH DATASETS

    # BREAST CANCER DATASET CLEANING
    #print("DIRTY DATA: \n" + breastCancerDataFrame.to_string())
    breastCancerNoId = breastCancerDataFrame.drop(columns=['Sample_code_number']) # remove ID column

    breastCancerClean = cleanData(breastCancerData, breastCancerNoId, False)
    breastCancerClean['Bare_nuclei'] = breastCancerClean['Bare_nuclei'].astype(int)
    # END BREAST CANCER DATASET CLEANING

    # GLASS DATASET CLEANING
    glassDataFrame = glassDataFrame.drop(columns=['Id_number'])
    glassClean = cleanData(glassData, glassDataFrame, False)
    # END GLASS DATASET CLEANING

    # IRIS DATASET CLEANING
    irisClean = cleanData(irisData, irisDataFrame, False)
    # END IRIS DATASET CLEANING

    # SOYBEAN DATASET CLEANING
    soybeanClean = cleanData(soybeanData, soybeanDataFrame, False)
    # END SOYBEAN DATASET CLEANING

    # VOTING DATASET CLEANING
    votingClean = cleanData(votingData, votingDataFrame, False)
    # END VOTING DATASET CLEANING

    #CROSS-VALIDATION, TRAINING + TESTING
    breastCancerFolds = crossValidation(breastCancerClean)
    breastCancerF1 = []
    breastCancerLoss = []

    #glassFolds = crossValidation(glassClean)

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

def cleanData(dataOriginal, dataFrame, noise):
    #dataVariables = pd.DataFrame(dataOriginal.variables)
    classColumnName = dataOriginal.variables.loc[dataOriginal.variables['role'] == 'Target', 'name'].values[0]
    columnTypes = dict(zip(dataOriginal.variables['name'], dataOriginal.variables['type']))

    # ADDRESS NULL VALUES WHERE COLUMNS/ROWS NEED TO BE REMOVED
    # If true class is unknown, drop the row
    dataFrame = dataFrame.dropna(subset=[classColumnName])
    # Drop any rows where all values are null
    dataRemovedNullRows = dataFrame.dropna(how = 'all')
    # Columns must have 70% of their values for rows to remain in dataset
    dataRemovedNullCols = dataRemovedNullRows.dropna(axis=1, thresh = math.floor(0.70*dataFrame.shape[0]))

    # ADD NOISE
    if(noise):
        addNoise(dataFrame, classColumnName)

    # BINNING OF CONTINUOUS/CATEGORICAL COLUMNS
    for columnName in dataFrame.columns:
        columnRole = dataOriginal.variables.loc[dataOriginal.variables['name'] == columnName, 'role'].values[0]

        # Ignore class column (target)
        if columnRole != 'Target':
            # if continuous, make categorical (5 categories total)
            if columnTypes[columnName] == 'Continuous':
                # split dataset into 5 equal-width bins
                binVals = np.linspace(dataFrame[columnName].min(), dataFrame[columnName].max(), 6)
                binLabels = ['A', 'B', 'C', 'D', 'E']

                # assign column vals to a bin
                dataFrame[columnName] = pd.cut(dataFrame[columnName], bins = binVals, labels = binLabels, include_lowest = True)
                # set 'type' to Categorical
                columnTypes[columnName] = 'Categorical'

            if columnTypes[columnName] == 'Categorical':
                dataFrame = dataRemovedNullCols.fillna(method='ffill')
            else:
                # fill na's with the rounded mean of the column (whole numbers will work w/ ints and floats)
                dataFrame = dataRemovedNullCols.fillna(round(dataRemovedNullCols.mean()))

    return dataFrame, classColumnName

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
        dataChunks[i] = chunk

        # rest of dataset without selected chunk
        tempDataset = tempDataset.drop(chunk.index)

    # the last chunk might be slightly different size if dataset size is not divisible by 10
    dataChunks[9] = tempDataset

    # rotate each part to be used as testing 1x
    # call learn, classify, etc. on each version of the train/test data
    return dataChunks
main()
