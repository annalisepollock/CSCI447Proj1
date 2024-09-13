import math
import random
import warnings
from types import NoneType

#from pandas.core.common import SettingWithCopyWarning
import pandas as pd
import numpy as np
import Learner
import AlgorithmAccuracy
import ClassificationInfo
from ucimlrepo import fetch_ucirepo

#warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

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
    breastCancerNoId = breastCancerDataFrame.drop(columns=['Sample_code_number']) # remove ID column

    
    breastCancerClean = cleanData(breastCancerData, breastCancerNoId, False)
    breastCancerNoise = cleanData(breastCancerData, breastCancerNoId, True)
    breastCancerClean[0]['Bare_nuclei'] = breastCancerClean[0]['Bare_nuclei'].astype(int)
    breastCancerNoise[0]['Bare_nuclei'] = breastCancerNoise[0]['Bare_nuclei'].astype(int)
    # END BREAST CANCER DATASET CLEANING

    # GLASS DATASET CLEANING
    glassDataFrame = glassDataFrame.drop(columns=['Id_number'])
    glassClean = cleanData(glassData, glassDataFrame, False)
    glassNoise = cleanData(glassData, glassDataFrame, True)
    # END GLASS DATASET CLEANING

    # IRIS DATASET CLEANING
    irisClean = cleanData(irisData, irisDataFrame, False)
    irisNoise = cleanData(irisData, irisDataFrame, True)
    # END IRIS DATASET CLEANING

    # SOYBEAN DATASET CLEANING
    soybeanClean = cleanData(soybeanData, soybeanDataFrame, False)
    soybeanNoise = cleanData(soybeanData, soybeanDataFrame, True)
    # END SOYBEAN DATASET CLEANING

    # VOTING DATASET CLEANING
    votingClean = cleanData(votingData, votingDataFrame, False)
    votingNoise = cleanData(votingData, votingDataFrame, True)
    # END VOTING DATASET CLEANING

    
    # CROSS-VALIDATION, TRAINING + TESTING
    breastCancerFolds = crossValidation(breastCancerClean[0], breastCancerClean[1])
    breastCancerNoiseFolds = crossValidation(breastCancerNoise[0], breastCancerNoise[1])

    breastCancerClassification = ClassificationInfo.ClassificationInfo()
    breastCancerNoiseClassification = ClassificationInfo.ClassificationInfo()

    print("Training Breast Cancer")
    for fold in breastCancerFolds:
        train = breastCancerClean[0].drop(fold.index)
        learner = Learner.Learner(train, breastCancerClean[1], breastCancerClassification)
        learner.classify(fold)
    print("Breast Cancer Accuracy:")
    breastCancerStats = AlgorithmAccuracy.AlgorithmAccuracy(breastCancerClassification)
    breastCancerStats.print()
    print()

    print("Training Breast Cancer Noise Data")
    for fold in breastCancerNoiseFolds:
        train = breastCancerNoise[0].drop(fold.index)
        learner = Learner.Learner(train, breastCancerClean[1], breastCancerNoiseClassification)
        learner.classify(fold)
    
    print("Breast Cancer Noise Accuracy")
    breastCancerNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(breastCancerNoiseClassification)
    breastCancerNoiseStats.print()
    print()
    
    #FINISHED BREAST CANCER DATASET
    
    glassFolds = crossValidation(glassClean[0], glassClean[1])
    glassNoiseFolds = crossValidation(glassNoise[0], glassNoise[1])

    glassClassification = ClassificationInfo.ClassificationInfo()
    glassNoiseClassification = ClassificationInfo.ClassificationInfo()

    print("Training Glass Data")
    for fold in glassFolds:
        train = glassClean[0].drop(fold.index)
        learner = Learner.Learner(train, glassClean[1], glassClassification)
        learner.classify(fold)
    
    print("Glass Data Accuracy")
    glassStats = AlgorithmAccuracy.AlgorithmAccuracy(glassClassification)
    glassStats.print()
    print()

    print("Training Glass Noise")
    for fold in glassNoiseFolds:
        train = glassNoise[0].drop(fold.index)
        learner = Learner.Learner(train, glassClean[1], glassNoiseClassification)
        learner.classify(fold)

    print("Glass Noise Accuracy")
    glassNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(glassNoiseClassification)
    glassNoiseStats.print()
    print()


    #FINISHED GLASS DATASET

    irisFolds = crossValidation(irisClean[0], irisClean[1])
    irisNoiseFolds = crossValidation(irisNoise[0], irisNoise[1])

    irisClassification = ClassificationInfo.ClassificationInfo()
    irisNoiseClassification = ClassificationInfo.ClassificationInfo()

    print("Training Iris Data")
    for fold in irisFolds:
        train = irisClean[0].drop(fold.index)
        learner = Learner.Learner(train, irisClean[1], irisClassification)
        learner.classify(fold)
    
    print("Iris Accuracy")
    irisStats = AlgorithmAccuracy.AlgorithmAccuracy(irisClassification)
    irisStats.print()
    print()

    print("Testing Iris Noise Data")
    for fold in irisNoiseFolds:
        train = irisNoise[0].drop(fold.index)
        learner = Learner.Learner(train, irisClean[1], irisNoiseClassification)
        learner.classify(fold)
    
    print("Iris Noise Accuracy")
    irisNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(irisNoiseClassification)
    irisNoiseStats.print()
    print()
    #FINISHED IRIS DATASET
    
    soybeanFolds = crossValidation(soybeanClean[0], soybeanClean[1])
    soybeanNoiseFolds = crossValidation(soybeanNoise[0], soybeanNoise[1])

    soybeanClassification = ClassificationInfo.ClassificationInfo()
    soybeanNoiseClassification = ClassificationInfo.ClassificationInfo()

    print("Training Soybean Data")
    for fold in soybeanFolds:
        train = soybeanClean[0].drop(fold.index)
        learner = Learner.Learner(train, soybeanClean[1], soybeanClassification)
        learner.classify(fold)

    print("Soybean Accuracy:")  
    soybeanStats = AlgorithmAccuracy.AlgorithmAccuracy(soybeanClassification)
    soybeanStats.print()
    print()

    print("Training Soybean Noise")
    for fold in soybeanNoiseFolds:
        train = soybeanNoise[0].drop(fold.index)
        learner = Learner.Learner(train, soybeanClean[1], soybeanNoiseClassification)
        learner.classify(fold)
    
    print("Soybean Noise Accuracy:")
    soybeanNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(soybeanNoiseClassification)
    soybeanNoiseStats.print()
    print()
    #FINISHED SOYBEAN DATASET

    votingFolds = crossValidation(votingClean[0], votingClean[1])
    votingNoiseFolds = crossValidation(votingNoise[0], votingNoise[1])

    votingClassification = ClassificationInfo.ClassificationInfo()
    votingNoiseClassification = ClassificationInfo.ClassificationInfo()

    print("Training Voting Data")
    for fold in votingFolds:
        train = votingClean[0].drop(fold.index)
        learner = Learner.Learner(train, 'Class', votingClassification)
        learner.classify(fold)
    
    print("Voting Accuracy:")
    votingStats = AlgorithmAccuracy.AlgorithmAccuracy(votingClassification)
    votingStats.print()
    print()

    print("Training Voting Noise")
    for fold in votingNoiseFolds:
        train = votingNoise[0].drop(fold.index)
        learner = Learner.Learner(train, 'Class', votingNoiseClassification)
        learner.classify(fold)

    print("Voting Noise Accuracy: ")
    votingNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(votingNoiseClassification)
    votingNoiseStats.print()
    print()
    
    #FINISHED VOTING DATASET

    # PLOTS
    import matplotlib.pyplot as plt
    import numpy as np

    # Categories for the x-axis
    categories = ['Noise', 'No Noise']

    # F1 score and 0-1 loss for each category
    breastCancerF1 = [breastCancerStats.getF1(), breastCancerNoiseStats.getF1()]  # Example values
    breastCancerLoss = [breastCancerStats.getLoss(), breastCancerNoiseStats.getLoss()]  # Example values

    # Set width for bars
    bar_width = 0.35

    # Create an array for the x-axis
    x = np.arange(len(categories))

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot F1 score bars
    bars_f1 = ax.bar(x - bar_width / 2, breastCancerF1, bar_width, label='F1 Score', color='blue')

    # Plot 0-1 loss bars
    bars_loss = ax.bar(x + bar_width / 2, breastCancerLoss, bar_width, label='0-1 Loss', color='orange')

    # Add labels, title, and legend
    ax.set_xlabel('Category')
    ax.set_ylabel('Scores')
    ax.set_title('Breast Cancer Accuracy States with Noise vs No Noise')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Display the plot
    plt.show()
    # END PLOTS
    f1BreastCancer = [['BreastCancer', breastCancerStats.getF1], ['BreastCancerNoise', breastCancerNoiseStats.getF1]]
    lossBreastCancer = [['BreastCancer', breastCancerStats.getLoss], ['BreastCancerNoise', breastCancerNoiseStats.getLoss]]

    f1Glass = [['Glass', glassStats.getF1], ['GlassNoise', glassNoiseStats.getF1]]
    lossGlass = [['Glass', glassStats.getLoss], ['GlassNoise', glassNoiseStats.getLoss]]

    f1Iris = [['Iris', irisStats.getF1], ['IrisNoise', irisNoiseStats.getF1]]
    lossIris = [['Iris', irisStats.getLoss], ['IrisNoise', irisNoiseStats.getLoss]]

    f1Soybean = [['Soybean', soybeanStats.getF1], ['SoybeanNoise', soybeanNoiseStats.getF1]]
    lossSoybean = [['Soybean', soybeanStats.getLoss], ['SoybeanNoise', soybeanNoiseStats.getLoss]]

    f1Voting = [['Voting', votingStats.getF1], ['VotingNoise', votingNoiseStats.getF1]]
    lossVoting = [['Voting', votingStats.getLoss], ['VotingNoise', votingNoiseStats.getLoss]]



def cleanData(dataOriginal, dataFrame, noise):
    #dataVariables = pd.DataFrame(dataOriginal.variables)
    classColumnName = dataOriginal.variables.loc[dataOriginal.variables['role'] == 'Target', 'name'].values[0]
    columnTypes = dict(zip(dataOriginal.variables['name'], dataOriginal.variables['type']))

    # ADDRESS NULL VALUES WHERE COLUMNS/ROWS NEED TO BE REMOVED
    # If true class is unknown, drop the row
    dataFrame = dataFrame.dropna(subset=[classColumnName])
    # Drop any rows where all values are null
    dataFrame = dataFrame.dropna(how = 'all')
    # Columns must have 70% of their values for rows to remain in dataset
    dataFrame = dataFrame.dropna(axis=1, thresh = math.floor(0.70*dataFrame.shape[0]))

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
                dataFrame = dataFrame.ffill()
            else:
                # fill na's with the rounded mean of the column (whole numbers will work w/ ints and floats)
                dataFrame = dataFrame.fillna(round(dataFrame[columnName].mean()))
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

def crossValidation(cleanDataset, classColumn):
    # 10-fold cross validation with stratification of classes
    print("Running Cross Validation with Stratification of Classes...")
    dataChunks = [None] * 10
    classes = np.unique(cleanDataset[classColumn])
    dataByClass = dict()

    for uniqueVal in classes:
        # Subset data based on unique class values
        classSubset = cleanDataset[cleanDataset[classColumn] == uniqueVal]
        print("Creating a subset of data for class " + str(uniqueVal) + " with size of " + str(classSubset.size))
        dataByClass[uniqueVal] = classSubset

        numRows = math.floor(classSubset.shape[0] / 10) # of class instances per fold

        for i in range(9):
            classChunk = classSubset.sample(n=numRows)
            print("Number of values for class " + str(uniqueVal), " in fold " + str(i+1) + " is: " + str(classChunk.shape[0]))
            if dataChunks[i] is None:
                dataChunks[i] = classChunk
            else:
                dataChunks[i] = pd.concat([dataChunks[i], classChunk])

            classSubset = classSubset.drop(classChunk.index)

        # the last chunk might be slightly different size if dataset size is not divisible by 10
        print("Number of values for class " + str(uniqueVal), " in fold " + str(10) + " is: " + str(classSubset.shape[0]))
        dataChunks[9] = pd.concat([dataChunks[9], classSubset])

    for i in range(len(dataChunks)):
        print("Size of fold " + str(i+1) + " is " + str(dataChunks[i].shape[0]))

    # for i in range(9):
    #     # randomly select 1/10 of the dataset, put it in the array
    #     chunk = tempDataset.sample(n=numRows)
    #     dataChunks[i] = chunk
    #
    #     # rest of dataset without selected chunk
    #     tempDataset = tempDataset.drop(chunk.index)
    #
    # # rotate each part to be used as testing 1x
    # # call learn, classify, etc. on each version of the train/test data
    return dataChunks
main()
