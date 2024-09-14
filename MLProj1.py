import math
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Learner
import AlgorithmAccuracy
import ClassificationInfo
from ucimlrepo import fetch_ucirepo


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

    print("FIRST 20 ROW OF IRIS DATA BEFORE CLEANING AND BINNING:")
    print(irisDataFrame.head(20))
    print()

    irisClean = cleanData(irisData, irisDataFrame, False)
    irisNoise = cleanData(irisData, irisDataFrame, True)

    print("FIRST 20 ROWS OF IRIS DATASET AFTER BINNING AND CLEANING:")
    print("NO NOISE:")
    print(irisClean[0].head(20))
    print()
    print("NOISE:")
    print(irisNoise[0].head(20))

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
    breastCancerFolds = crossValidation(breastCancerClean[0], breastCancerClean[1], False)
    breastCancerNoiseFolds = crossValidation(breastCancerNoise[0], breastCancerNoise[1], False)

    breastCancerClassification = ClassificationInfo.ClassificationInfo()
    breastCancerNoiseClassification = ClassificationInfo.ClassificationInfo()

    breastCancerFoldsAccuracy = []
    breastCancerNoiseFoldsAccuracy = []

    for fold in breastCancerFolds:
        train = breastCancerClean[0].drop(fold.index)
        learner = Learner.Learner(train, breastCancerClean[1], breastCancerClassification)
        breastCancerFoldsAccuracy.append(learner.classify(fold))

    breastCancerStats = AlgorithmAccuracy.AlgorithmAccuracy(breastCancerClassification, len(breastCancerClean[0].columns), "Breast Cancer")

    for fold in breastCancerNoiseFolds:
        train = breastCancerNoise[0].drop(fold.index)
        learner = Learner.Learner(train, breastCancerClean[1], breastCancerNoiseClassification)
        breastCancerNoiseFoldsAccuracy.append(learner.classify(fold))

    breastCancerNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(breastCancerNoiseClassification, len(breastCancerClean[0].columns), "Breast Cancer Noise")
    #FINISHED BREAST CANCER DATASET
    
    glassFolds = crossValidation(glassClean[0], glassClean[1], False)
    glassNoiseFolds = crossValidation(glassNoise[0], glassNoise[1], False)

    glassClassification = ClassificationInfo.ClassificationInfo()
    glassNoiseClassification = ClassificationInfo.ClassificationInfo()

    glassFoldsAccuracy = []
    glassNoiseFoldsAccuracy = []


    for fold in glassFolds:
        train = glassClean[0].drop(fold.index)
        learner = Learner.Learner(train, glassClean[1], glassClassification)
        glassFoldsAccuracy.append(learner.classify(fold))

    glassStats = AlgorithmAccuracy.AlgorithmAccuracy(glassClassification, len(glassClean[0].columns), "Glass")

    for fold in glassNoiseFolds:
        train = glassNoise[0].drop(fold.index)
        learner = Learner.Learner(train, glassClean[1], glassNoiseClassification)
        glassNoiseFoldsAccuracy.append(learner.classify(fold))

    glassNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(glassNoiseClassification, len(glassClean[0].columns), "Glass Noise")

    #FINISHED GLASS DATASET

    print("CROSS VALIDATION FOR IRIS (NO NOISE)")
    irisFolds = crossValidation(irisClean[0], irisClean[1], True)
    print("CROSS VALIDATION FOR IRIS (NOISE)")
    irisNoiseFolds = crossValidation(irisNoise[0], irisNoise[1], True)

    irisClassification = ClassificationInfo.ClassificationInfo()
    irisNoiseClassification = ClassificationInfo.ClassificationInfo()

    irisFoldsAccuracy = []
    irisNoiseFoldsAccuracy = []

    print("\nCLASSIFYING IRIS DATA")
    count = 0 
    for fold in irisFolds:
        train = irisClean[0].drop(fold.index)
        learner = Learner.Learner(train, irisClean[1], irisClassification)
        if(count == 0):
            learner.printInfo()
            print()
            print("CLASSIFICATION DEMONSTRATION FOR ONE FOLD:")
        irisFoldsAccuracy.append(learner.classify(fold, count == 0))
        count += 1
    
    irisStats = AlgorithmAccuracy.AlgorithmAccuracy(irisClassification, len(irisClean[0].columns), "Iris")

    for fold in irisNoiseFolds:
        train = irisNoise[0].drop(fold.index)
        learner = Learner.Learner(train, irisClean[1], irisNoiseClassification)
        irisNoiseFoldsAccuracy.append(learner.classify(fold))

    irisNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(irisNoiseClassification, len(irisClean[0].columns), "Iris Noise")

    #FINISHED IRIS DATASET
    
    soybeanFolds = crossValidation(soybeanClean[0], soybeanClean[1], False)
    soybeanNoiseFolds = crossValidation(soybeanNoise[0], soybeanNoise[1], False)

    soybeanClassification = ClassificationInfo.ClassificationInfo()
    soybeanNoiseClassification = ClassificationInfo.ClassificationInfo()

    soybeanFoldsAccuracy = []
    soybeanNoiseFoldsAccuracy = []

    for fold in soybeanFolds:
        train = soybeanClean[0].drop(fold.index)
        learner = Learner.Learner(train, soybeanClean[1], soybeanClassification)
        soybeanFoldsAccuracy.append(learner.classify(fold))

    soybeanStats = AlgorithmAccuracy.AlgorithmAccuracy(soybeanClassification, len(soybeanClean[0].columns), "Soybean")
   
    for fold in soybeanNoiseFolds:
        train = soybeanNoise[0].drop(fold.index)
        learner = Learner.Learner(train, soybeanClean[1], soybeanNoiseClassification)
        soybeanNoiseFoldsAccuracy.append(learner.classify(fold))
    
    soybeanNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(soybeanNoiseClassification, len(soybeanClean[0].columns), "Soybean Noise")
    #FINISHED SOYBEAN DATASET

    votingFolds = crossValidation(votingClean[0], votingClean[1], False)
    votingNoiseFolds = crossValidation(votingNoise[0], votingNoise[1], False)

    votingClassification = ClassificationInfo.ClassificationInfo()
    votingNoiseClassification = ClassificationInfo.ClassificationInfo()

    votingFoldsAccuracy = []
    votingNoiseFoldsAccuracy = []

    for fold in votingFolds:
        train = votingClean[0].drop(fold.index)
        learner = Learner.Learner(train, 'Class', votingClassification)
        votingFoldsAccuracy.append(learner.classify(fold))
    votingStats = AlgorithmAccuracy.AlgorithmAccuracy(votingClassification, len(votingClean[0].columns), "Voting")

    for fold in votingNoiseFolds:
        train = votingNoise[0].drop(fold.index)
        learner = Learner.Learner(train, 'Class', votingNoiseClassification)
        votingNoiseFoldsAccuracy.append(learner.classify(fold))

    votingNoiseStats = AlgorithmAccuracy.AlgorithmAccuracy(votingNoiseClassification, len(votingClean[0].columns), "Voting Noise")
 
    #FINISHED VOTING DATASET
    # PLOT DATA
    plotData("Breast Cancer", breastCancerStats, breastCancerNoiseStats, breastCancerFoldsAccuracy, breastCancerNoiseFoldsAccuracy)
    plotData("Glass", glassStats, glassNoiseStats, glassFoldsAccuracy, glassNoiseFoldsAccuracy)
    plotData("Iris", irisStats, irisNoiseStats, irisFoldsAccuracy, irisNoiseFoldsAccuracy)
    plotData("Soybean", soybeanStats, soybeanNoiseStats, soybeanFoldsAccuracy, soybeanNoiseFoldsAccuracy)
    plotData("Voting", votingStats, votingNoiseStats, votingFoldsAccuracy, votingNoiseFoldsAccuracy)
    # END PLOT DATA

def plotData(datasetName, noNoiseStats, noiseStats, noNoiseAccuracy, NoiseAccuracy):
    # Categories for the x-axis
    categories = ['Noise', 'No Noise']

    # F1 score and 0-1 loss for each category
    dataF1 = [noNoiseStats.getF1(), noiseStats.getF1()]  # Example values
    dataLoss = [noNoiseStats.getLoss(), noiseStats.getLoss()]  # Example values

    # Set width for bars
    bar_width = 0.35

    # Create an array for the x-axis
    x = np.arange(len(categories))

    # Create the figure and axis
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))

    # Plot F1 score bars
    barsF1 = ax[0].bar(x - bar_width / 2, dataF1, bar_width, label='F1 Score', color='blue')

    # Plot 0-1 loss bars
    barsLoss = ax[0].bar(x + bar_width / 2, dataLoss, bar_width, label='0-1 Loss', color='orange')

    # Add labels, title, and legend
    ax[0].set_xlabel('Noise')
    ax[0].set_ylabel('Scores')
    ax[0].set_title(datasetName + ' Accuracy Stats with Noise vs No Noise\n(calculated using totals across 10 folds)')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(categories)
    ax[0].legend()

    test = pd.DataFrame(noNoiseAccuracy)
    print(test.to_string())

    lossDataFrame = pd.DataFrame({
        'Accuracy': noNoiseAccuracy + NoiseAccuracy,
        'Category': ['No Noise'] * len(noNoiseAccuracy) + ['Noise'] * len(NoiseAccuracy)
    })
    # Plot the box plots
    sns.boxplot(data=lossDataFrame, x='Category', y='Accuracy', ax=ax[1], color='orange')

    # Overlay the actual data points with jitter for better visibility
    sns.stripplot(data=lossDataFrame, x='Category', y='Accuracy', ax=ax[1], color='black', jitter=False)

    # Add labels and title
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Box Plots of Loss for ' + datasetName + ' Classification')

    # Display the plot
    plt.tight_layout()

    plt.show()
    # END PLOTS

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

    if numCols < 1:
        numCols = 1
    #dont mix class column
    columns = dataSet.columns.drop(classColumnName).to_list()
    # Randomly select 10% of the columns
    selectedColumns = random.sample(columns, numCols)
    #grab data from those columns
    for col in selectedColumns:
        newCol = np.random.permutation(dataSet[col].values)
        dataSet[col] = newCol

    return dataSet

def crossValidation(cleanDataset, classColumn, printSteps):
    # 10-fold cross validation with stratification of classes
    if printSteps == True:
        print("Running cross calidation with stratification...")
    dataChunks = [None] * 10
    classes = np.unique(cleanDataset[classColumn])
    dataByClass = dict()

    for uniqueVal in classes:
        # Subset data based on unique class values
        classSubset = cleanDataset[cleanDataset[classColumn] == uniqueVal]
        if printSteps == True:
            print("Creating a subset of data for class " + str(uniqueVal) + " with size of " + str(classSubset.size))
        dataByClass[uniqueVal] = classSubset

        numRows = math.floor(classSubset.shape[0] / 10) # of class instances per fold

        for i in range(9):
            classChunk = classSubset.sample(n=numRows)
            if printSteps:
                print("Number of values for class " + str(uniqueVal), " in fold " + str(i+1) + " is: " + str(classChunk.shape[0]))
            if dataChunks[i] is None:
                dataChunks[i] = classChunk
            else:
                dataChunks[i] = pd.concat([dataChunks[i], classChunk])

            classSubset = classSubset.drop(classChunk.index)

        # the last chunk might be slightly different size if dataset size is not divisible by 10
        if printSteps == True:
            print("Number of values for class " + str(uniqueVal), " in fold " + str(10) + " is: " + str(classSubset.shape[0]))
        dataChunks[9] = pd.concat([dataChunks[9], classSubset])

    if printSteps == True:
        for i in range(len(dataChunks)):
            print("Size of fold " + str(i+1) + " is " + str(dataChunks[i].shape[0]))

    return dataChunks
main()
