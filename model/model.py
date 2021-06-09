"""
Dota 2 turtling hero prediction
Written for tensorflow 2.3.0

Written by Jasper Law

Pass a path to your data in argv. Can be a single CSV, or can be a path to a folder containing CSV(s)
Will not work for CSVs with less than 10 heroes. Training data must have isTurtling0-isTurtling9 label columns
"""

##### IMPORTS #####
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from pandas import read_csv, DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sys
import os
import numpy
import random
import glob
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType
from typing import Any, Dict, List
import subprocess as sp
import os
from datetime import timedelta
import time
import functools, itertools
import matplotlib
from matplotlib import pyplot as plt

# Uncomment for reproducable results:
# numpy.random.seed(50)
# tf.random.set_seed(50)


# tf.logging.set_verbosity(tf.logging.ERROR)

# Required for USE_DATASET, but doesnt work
# strategy = tf.distribute.MirroredStrategy()
# print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Uncomment to print where operations are computed:
tf.debugging.set_log_device_placement(True)


# Use GPU or CPU?
USE_GPU = False
# Minimum memory that a GPU must have free (in MiB) in order for it to be used.
# Don't want to hog all the system resources!
GPU_MEMORY_REQUIREMENT = 2048
# If not enough GPUs can be found, wait for this amount of time and try again.
GPU_WAIT_PERIOD = timedelta(minutes=10)
# Maximum number of GPUs to use in parallel
# May misbehave or error out when trying to test hyperparameter sets longer than this (unless you give 1, thats fine)
NUM_PARALLEL_GPUS = 8


# make a new model, or load from file? "load" or "new"
MODEL_LOADER = "new"
# When MODEL_LOADER is "load", use {directory containing training data}/models/MODEL_TO_LOAD as a keras model and load it
MODEL_TO_LOAD = "05-20-2021_08-25"
# "evaluate"=train and test. "plot"=draw a picture of the model. "hyperparameters"=to optimize hyperparams with linear search. "data-stats"=get number of turtles in dataset. "plot_weights"=Train a model, and plot how the weights of each neuron changed throughout. Great for seeing if the model is learning or not.
ACTION = "evaluate"
# If ACTION involves some evaluation (e.g "evaluate", or every step of "hyperparameters"), perform and save the results of a batch predict over the test dataset
EVALUATE = True
# If EVALUATE, save the data points, labels and predictions to CSV file for inspection.
SAVE_EVALUATION = True
# If EVALUATE, only evaluate this amount of the training set. Much quicker. E.g 0.001 for 0.1% of training data
POST_EVAL_PERCENT = 0.001
# Save the model to directory containing training data}/models/{current date-time}
SAVE_MODEL = False
# BROKEN. Transform into a tf.dataset for data parallelism
USE_TF_DATASET = False
# Whether to ensure train set has exactly TRAIN_POSITIVE_POINTS_PERCENT data points with at least one positive label, or to rely on randomness
ENSURE_TRAIN_POSITIVES = False
# Amount of datapoints with at least one positive label to use for training
TRAIN_POSITIVE_POINTS_PERCENT = 0.95

# When running on CPU, how many threads to use? -1 for max
NUM_THREADS = -1

## DATA PARAMETERS
# How much data to use for test/validate? Will be split in half for validate, i.e 0.2 for 80% train, 10% test, 10% validate
TEST_SPLIT = 0.2
# probability of data point removal %x100 i.e 50 for half of data points drop out. give -1 to make classes equal
UNDERSAMPLING_CHANCE = -1
# Do undersampling?
DO_UNDERSAMPLING = False
# Randomize which hero's data gets sent to which subnet?
DO_HERO_SHUFFLING = True
# Weight classes for balancing?
DO_CLASS_WEIGHTING = True

### Model parameters - not used during action="hyperparameters"
# learning rate
LEARNING_RATE = 5e-05#1e-06
# depth of the shared subnet
SHARED_NET_DEPTH = 64#29
# depth of the final subnet
FINAL_NET_DEPTH = 64#3
# batch size
BATCH_SIZE = 32
# Probability of neuron dropout
NN_DROPOUT = 0.1
# max weights norm enforced on layers just before neuron dropout
WEIGHT_CONSTRAIN = 1
# epochs
NUM_EPOCHS = 16
# don't touch. This will contain the directory containing your training data.
DATADIR = ""

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def getAllGPUMemorys():
    """List available memory for all nvidia GPUs"""
    # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for x in memory_free_info]
    return memory_free_values

def getGPUMemory(gpuNum):
    """Get available memory for a specific GPU"""
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv --id=" + str(gpuNum)
    memory_free_info = sp.check_output(COMMAND.split()).decode('ascii').split("\n")[1]
    return int(memory_free_info.rstrip(" MiB"))

def pickLeastUsedGPUs(numGPUs=3):
    """pick the numGPUs least use GPUs and return their device numbers in a list.
    If not enough GPUs can be found that have at least GPU_MEMORY_REQUIREMENT memory free,
    then GPU_WAIT_PERIOD will be waited, and the process will be retried, continuously.
    """
    allMems = getAllGPUMemorys()
    try:
        working = numpy.array(allMems)
        leastUsedIndices = numpy.argpartition(working, numGPUs)
        leastUsed = list(working[leastUsedIndices[-numGPUs:]])
    except ValueError as e:
        raise ValueError("Not enough GPUs in the system.\nOriginal exception: " + str(e))
    numUseable = len(list(i for i in leastUsed if i > GPU_MEMORY_REQUIREMENT))
    timesSlept = 0
    while numUseable < numGPUs:
        timesSlept += 1
        print("Not enough GPUs with enough free memory, sleeping for " + str(GPU_WAIT_PERIOD) + "(sleep number " + str(timesSlept) + ")")
        time.sleep(GPU_WAIT_PERIOD.total_seconds())
        allMems = getAllGPUMemorys()
        try:
            working = numpy.array(allMems)
            leastUsedIndices = numpy.argpartition(working, numGPUs)
            leastUsed = list(working[leastUsedIndices[-numGPUs:]])
        except ValueError as e:
            raise ValueError("Not enough GPUs in the system.\nOriginal exception: " + str(e))
        numUseable = len(list(i for i in leastUsed if i > GPU_MEMORY_REQUIREMENT))
    
    gpuIndices = [leastUsedIndices[i] for i in range(len(allMems) - 1, len(allMems) - numGPUs - 1, -1)]
    print("least used GPUs selected: " + ", ".join(str(i) + " (" + str(allMems[i]) + ")" for i in gpuIndices))
    print("allMems: " + ", ".join(str(i) for i in allMems))
    return gpuIndices

# Generate list of headers for EVALUATE saving. HARD CODED!
dataHeaders = ["tick"]
for heroNum in range(10):
    dataHeaders += [f"heroID{heroNum}", f"heroTeam{heroNum}", f"posX{heroNum}", f"posXPerSecond{heroNum}", f"posY{heroNum}", f"posYPerSecond{heroNum}", f"posZ{heroNum}", f"posZPerSecond{heroNum}", f"netWorth{heroNum}", f"netWorthPerSecond{heroNum}", f"XP{heroNum}", f"XPPerSecond{heroNum}", f"kills{heroNum}", f"deaths{heroNum}", f"lastHits{heroNum}", f"lastHitsPerSecond{heroNum}", f"closestFriendlyHeroDist{heroNum}", f"closestEnemyHeroDist{heroNum}", f"closestFriendlyTowerDist{heroNum}", f"closestEnemyTowerDist{heroNum}"]
for heroNum in range(10):
    dataHeaders += [f"isTurtling{heroNum}"]
for heroNum in range(10):
    dataHeaders += [f"predictedIsTurtling{heroNum}"]


if USE_GPU:
    # Make matplotlib headless
    matplotlib.use("Agg")
    # initialize GPUs
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


print("IMPORT COMPLETE")

def numExtFName(baseFName, ext):
    """Take baseFName, Make any missing intermediary folders, and add an extension to the filename so that it doesnt clash with existing files. e.g myFile.txt -> myFile (1).txt"""
    dirPath = os.path.dirname(baseFName + ext)
    if dirPath and not os.path.exists(dirPath):
        os.makedirs(dirPath)
    outpath = baseFName + ext
    if os.path.isfile(outpath):
        modifier = 0
        while os.path.isfile(outpath):
            modifier += 1
            outpath = baseFName + " (" + str(modifier) + ")" + ext
    return outpath


# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
early_stopping = keras.callbacks.EarlyStopping(
    monitor='prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

def plot_metrics(history):
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        # plt.plot(history.epoch, history.history['val_'+metric],
        #             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()


def singleHyperParamSearch(modelFactory: FunctionType, staticParams: Dict[str, Any], searchParamName: str, searchParamValues: list, trainX, trainY, testX, testY, validateX, validateY, gpuIDs):
    """Find the locally optimal hyperparam configuration through linear search
    this is not grid search, it only searches values for a single param
    """
    results = {}
    if USE_GPU:
        gpuParams = [{"gpuName": "/GPU:" + str(gpuIDs[min(i, len(gpuIDs)-1)]), "paramValue": searchParamValues[i]} for i in range(len(searchParamValues))]
    else:
        gpuParams = [{"gpuName": "/CPU:" + str(gpuIDs[min(i, len(gpuIDs)-1)]), "paramValue": searchParamValues[i]} for i in range(len(searchParamValues))]
    staticParams = staticParams.copy()
    if searchParamName in staticParams:
        del staticParams[searchParamName]

    def tryModel(currentParams):
        paramValue, gpuName = currentParams["paramValue"], currentParams["gpuName"]
        batchSize = paramValue if searchParamName == "batch_size" else staticParams.get("batch_size", BATCH_SIZE) 
        epochs = paramValue if searchParamName == "epochs" else staticParams.get("epochs", NUM_EPOCHS) 
        
        if searchParamName in ("batch_size", "epochs"):
            factoryArgs = staticParams
        else:
            factoryArgs = {**staticParams, **{searchParamName: paramValue}}

        for p in ("batch_size", "epochs"):
            if p in factoryArgs:
                del factoryArgs[p]

        print("Trying model for " + searchParamName + "=" + str(paramValue))
        try:
            with tf.device(gpuName):
                
                model = modelFactory(**factoryArgs)

                model.fit(x=trainX, y=trainY, batch_size=batchSize, epochs=epochs, verbose=2, shuffle=True, callbacks=[early_stopping])#, class_weight=class_weight)#, validation_data=(validateX, validateY))#validation_split=0.5, 
                currentResults = model.evaluate(x=testX, y=testY)

                print(searchParamName + "=" + str(paramValue) + " complete, loss: %.4f" % currentResults)
                results[paramValue] = currentResults

                if EVALUATE:
                    print("##!## begin evaluation")
                    # predictInputs = numpy.ndarray(shape=(validateX.shape[0], 10*validateX.shape[1]))
                    predictionsData = numpy.ndarray(shape=(validateX.shape[0], 10*validateX.shape[1] + testY.shape[1]*2))
                    for heroNum in range(10):
                        # print("validateX[:, :, " + str(heroNum) + "] = " + str(validateX[:, :, heroNum].shape))
                        # print("predictionsData[:, " + str(heroNum*validateX.shape[1]) + ":" + str((1+heroNum)*validateX.shape[1]) + "] = " + str(validateX[:, :, heroNum].shape))
                        # predictInputs[:, heroNum*validateX.shape[1]:(1+heroNum)*validateX.shape[1]] = validateX[:, :, heroNum]
                        predictionsData[:, heroNum*validateX.shape[1]:(1+heroNum)*validateX.shape[1]] = validateX[:, :, heroNum]
                    # predictionsData[: validateX.shape[1]:validateY.shape[1]] = validateY

                    print("validateX: " + str(validateX.shape))

                    for testNum in range(validateX.shape[0]):
                        if testNum % 100 == 0:
                            print(str(int((testNum / validateX.shape[0]) * 100)) + "% Predicting test " + str(testNum) + "/" + str(validateX.shape[0]-1))
                        predictionsData[testNum, 10*validateX.shape[1]+validateY.shape[1]:] = model.predict(numpy.asarray([validateX[testNum, :, :]]))

                    outpath = numExtFName(DATADIR + os.sep + "hyperparam-search" + os.sep + searchParamName + os.sep + str(currentParams["paramValue"]) + os.sep + "predictions", ".csv")
                    DataFrame(predictionsData, columns=dataHeaders).to_csv(outpath, index=False)
                    # numpy.savetxt(outpath, predictionsData, delimiter=",")
                    print("evaluation complete, results saved to",outpath)

        except Exception as e:
            print(e)
            exit()
            raise e
    
    with ThreadPoolExecutor() as executor:
        executor.map(tryModel, gpuParams)
    
    return results


def singleHyperParamSearch_inputAsDataset(modelFactory: FunctionType, staticParams: Dict[str, Any], searchParamName: str, searchParamValues: list, trainData, testData, validateData, trainDataShapes, testDataShapes, validateDataShapes, gpuIDs):
    """copy of singleHyperParamSearch that expects train, test and validate as tf Datasets rather than two numpy arrays each
    data shape args should be tuple of (x shape, y shape)"""
    results = {}
    gpuParams = [{"gpuName": "/GPU:" + str(gpuIDs[i]), "paramValue": searchParamValues[i]} for i in range(len(searchParamValues))]
    staticParams = staticParams.copy()
    if searchParamName in staticParams:
        del staticParams[searchParamName]

    def tryModel(currentParams):
        paramValue, gpuName = currentParams["paramValue"], currentParams["gpuName"]
        batchSize = paramValue if searchParamName == "batch_size" else staticParams.get("batch_size", BATCH_SIZE) 
        epochs = paramValue if searchParamName == "epochs" else staticParams.get("epochs", NUM_EPOCHS) 
        
        if searchParamName in ("batch_size", "epochs"):
            factoryArgs = staticParams
        else:
            factoryArgs = {**staticParams, **{searchParamName: paramValue}}

        for p in ("batch_size", "epochs"):
            if p in factoryArgs:
                del factoryArgs[p]

        print("Trying model for " + searchParamName + "=" + str(paramValue))
        try:
            with tf.device(gpuName):
                
                model = modelFactory(**factoryArgs)

                model.fit(trainData, batch_size=batchSize, epochs=epochs, verbose=2, shuffle=True, callbacks=[early_stopping])#, class_weight=class_weight)#, validation_data=validateData)
                currentResults = model.evaluate(testData)

                print(searchParamName + "=" + str(paramValue) + " complete, loss: %.4f" % currentResults)
                results[paramValue] = currentResults

                if EVALUATE:
                    print("##!## begin evaluation")
                    heroColumns = nonHeroVariables + (variablesPerHero) * 10
                    predictionsShape = (validateDataShapes[0][0], heroColumns + validateDataShapes[1][1] + 10)
                    predictionsData = numpy.ndarray(shape=predictionsShape)
                    predictionsData[:, :nonHeroVariables] = heroTestData["hero0"][:, :nonHeroVariables]
                    # for colName, colNum in headers.items():
                    #     predictionsData[0, colNum] = colName

                    # for colNum in range(10):
                    #     predictionsData[0, -(10-colNum)] = "predictedIsTurtling" + str(colNum)

                    predictionsData[:,:heroIDStartIndices[0]] = heroTestData["hero0"][:, :nonHeroVariables]
                    for hero in heroTestData:
                        heroNum = int(hero[len("hero"):])
                        predictionsData[:,heroIDStartIndices[heroNum]:heroIDStartIndices[heroNum] + variablesPerHero] = heroTestData[hero][:, nonHeroVariables:]
                    # predictionsData[: predictionsShape[0], :testX.shape[1]] = heroTestData
                    predictionsData[:,heroColumns:heroColumns + testY.shape[1]] = shuffledTestY
                    predictionsData[:,-10:] = model.predict(testData_asDataset)


                    

                    # predictInputs = numpy.ndarray(shape=(validateX.shape[0], 10*validateX.shape[1]))
                    predictionsData = numpy.ndarray(shape=(validateDataShapes[0][0], 10*validateDataShapes[0][1] + validateDataShapes[1][1]*2))
                    # for heroNum in range(10):
                        # print("validateX[:, :, " + str(heroNum) + "] = " + str(validateX[:, :, heroNum].shape))
                        # print("predictionsData[:, " + str(heroNum*validateX.shape[1]) + ":" + str((1+heroNum)*validateX.shape[1]) + "] = " + str(validateX[:, :, heroNum].shape))
                        # predictInputs[:, heroNum*validateX.shape[1]:(1+heroNum)*validateX.shape[1]] = validateX[:, :, heroNum]
                        # predictionsData[:, heroNum*validateDataShapes[0][1]:(1+heroNum)*validateDataShapes[0][1]] = validateX[:, :, heroNum]
                    # predictionsData[: validateX.shape[1]:validateY.shape[1]] = validateY

                    print("validateX: " + str(validateDataShapes[0]))

                    testNum = 0
                    for currentTestInput, currentExpectedOutput in validateData.as_numpy_iterator():
                        if testNum % 100 == 0:
                            print(str(int((testNum / validateDataShapes[0][0]) * 100)) + "% Predicting test " + str(testNum) + "/" + str(validateDataShapes[0][0]-1))

                        predictionsData[testNum, :-2*validateDataShapes[1][1]] = currentTestInput
                        predictionsData[testNum, -2*validateDataShapes[1][1]:-validateDataShapes[1][1]] = currentExpectedOutput
                        predictionsData[testNum, -validateDataShapes[1][1]:] = model.predict(numpy.asarray([currentTestInput]))
                        testNum += 1

                    outpath = numExtFName(DATADIR + os.sep + "hyperparam-search" + os.sep + searchParamName + os.sep + str(currentParams["paramValue"]) + os.sep + "predictions", ".csv")
                    DataFrame(predictionsData, columns=dataHeaders).to_csv(outpath, index=False)
                    # numpy.savetxt(outpath, predictionsData, delimiter=",")
                    print("evaluation complete, results saved to",outpath)

        except Exception as e:
            print(e)
            exit()
            raise e
    
    with ThreadPoolExecutor() as executor:
        executor.map(tryModel, gpuParams)
    
    return results


##### READ DATA #####

if len(sys.argv) > 1:
    DATADIR = os.path.dirname(sys.argv[1])
    if not os.path.isdir(DATADIR):
        os.makedirs(DATADIR)

    if os.path.isfile(sys.argv[1]):
        filePaths = [os.path.normpath(p) for p in sys.argv[1:]]
    elif os.path.isdir(sys.argv[1]):
        if len(sys.argv) > 2:
            print("First argument is a directory, further arguments are ignored. First argument: '" + sys.argv[1] + "'")
        files = glob.glob(os.path.join(sys.argv[1],"*.csv"))
        if not files:
            print("no .csv files in the given folder")
        elif input(f"Train on {len(files)} csv files? (y/n)\n").lower() != "y":
            print("batch predict cancelled.")
            exit()
        else:
            filePaths = [os.path.normpath(p) for p in files]
    else:
        raise FileNotFoundError("Unrecognised path: '" + sys.argv[1] + "'")
else:
    raise ValueError("Please provide paths to your csvs")

print("Reading CSV(s)...")
allData = [read_csv(fp, header=0) for fp in filePaths]
# for fNum, fName in enumerate(filePaths):
#     print(fName,allData[fNum].shape[1])
print("CSV read(s) complete, begin validation of CSV(s)")

turtleHeroes = 0
nonTurtleHeroes = 0

if ACTION == "data-stats":
    ticksThatHaveTurtles = 0
    ticksThatDontHaveTurtles = 0

if len(allData) == 1:
    dataframe = allData[0]
    for heroNum in range(10):
        if f"isTurtling{heroNum!s}" not in dataframe.columns:
            raise ValueError(f"Missing target column: isTurtling{heroNum!s}")
        # Convert hero IDs to 1 hot AND NORMALIZE
        dataframe[f"heroID{heroNum!s}"] = (2 ** dataframe[f"heroID{heroNum!s}"][0]) / (2 ** 129)
        # Convert teams to 1 hot
        dataframe[f"heroTeam{heroNum!s}"] = {"dire": 0., "radiant": 1.}[dataframe[f"heroTeam{heroNum!s}"][0]]

    dataframe.sort_values("tick", inplace=True)
    # offset ticks back to zero-based
    # preRange = (max(dataframe["tick"]), min(dataframe["tick"]))
    dataframe["tick"] -= dataframe["tick"][0]
    # print(os.path.basename(filePaths[0]) + ": " + str(preRange) + " reduced to: " + str((max(dataframe["tick"]), min(dataframe["tick"]))))
    
    currentFrameTurtles = len(list(_ for _ in (dataframe[f"isTurtling{heroNum!s}"][0] for heroNum in range(10) if dataframe[f"isTurtling{heroNum!s}"][0] == 1)))
    turtleHeroes += currentFrameTurtles
    nonTurtleHeroes += 10 - currentFrameTurtles
    print(dataframe.shape[0],"datapoints loaded from",os.path.basename(filePaths[0]),f"({currentFrameTurtles} turtles detected)" if currentFrameTurtles else "")

else:
    for dfNum, df in enumerate(allData[1:]):
        if df.shape[1] != allData[0].shape[1]:
            raise ValueError(filePaths[dfNum] + f" has a different number of columns ({df.shape[1]}) to " + filePaths[0] + f" ({allData[0].shape[1]})")

    newDfShape = (sum(df.shape[0] for df in allData), allData[0].shape[1])
    dataframe = DataFrame(numpy.zeros(shape=newDfShape), dtype=object)
    dataframe.columns = allData[0].columns
    currentRow = 0

    for dfNum, df in enumerate(allData):
        for heroNum in range(10):
            if f"isTurtling{heroNum!s}" not in df.columns:
                raise ValueError(f"Missing target column: isTurtling{heroNum!s}. File: {filePaths[dfNum]}")
            # Convert hero IDs to 1 hot AND NORMALIZE
            df[f"heroID{heroNum!s}"] = (2 ** df[f"heroID{heroNum!s}"][0]) / (2 ** 129)
            # Convert teams to 1 hot
            df[f"heroTeam{heroNum!s}"] = {"dire": 0., "radiant": 1.}[df[f"heroTeam{heroNum!s}"][0]]

        df.sort_values("tick", inplace=True)
        # offset ticks back to zero-based
        # preRange = (max(df["tick"]), min(df["tick"]))
        df["tick"] -= df["tick"][0]
        # print(os.path.basename(filePaths[dfNum]) + ": " + str(preRange) + " reduced to: " + str((max(df["tick"]), min(df["tick"]))))

        dataframe[currentRow:currentRow+df.shape[0]] = df
        currentRow += df.shape[0]

        if ACTION == "data-stats":
            currentTicksThatDontHaveTurtles = len(list(t for _, t in df.iterrows() if 1 not in t[[f"isTurtling{heroNum}" for heroNum in range(10)]].values))
            ticksThatDontHaveTurtles += currentTicksThatDontHaveTurtles
            ticksThatHaveTurtles += len(df) - currentTicksThatDontHaveTurtles

        currentFrameTurtles = len(list(_ for _ in (allData[dfNum][f"isTurtling{heroNum!s}"][0] for heroNum in range(10) if allData[dfNum][f"isTurtling{heroNum!s}"][0] == 1)))
        turtleHeroes += currentFrameTurtles
        nonTurtleHeroes += 10 - currentFrameTurtles
        print(allData[dfNum].shape[0],"datapoints loaded from",os.path.basename(filePaths[dfNum]),f"({currentFrameTurtles} turtles detected)" if currentFrameTurtles else "")

print(turtleHeroes,"turtling heroes detected",f"{nonTurtleHeroes} non-turtling heroes detected" if turtleHeroes else "")
print(dataframe.shape[0],"total datapoints loaded, begin preprocessing")

if ACTION == "data-stats":
    raise ValueError(f"ticksThatDontHaveTurtles {ticksThatDontHaveTurtles}, ticksThatHaveTurtles {ticksThatHaveTurtles}")


# figures for whole 37-match dataset
# POS_ITEMS = 333605
# NEG_ITEMS = 595195

labelCounts = [numpy.bincount(dataframe[f"isTurtling{heroNum}"]) for heroNum in range(10)]
NEG_ITEMS, POS_ITEMS = sum([i[0] for i in labelCounts]), sum([i[1] if len(i) > 1 else 0 for i in labelCounts])
TOTAL_ITEMS = POS_ITEMS + NEG_ITEMS
INITIAL_OUTPUT_BIAS = numpy.log([POS_ITEMS/NEG_ITEMS])
weight_for_0 = (1 / NEG_ITEMS) * (TOTAL_ITEMS / 2.0)#(TOTAL_ITEMS/2) #
weight_for_1 = (1 / POS_ITEMS) * (TOTAL_ITEMS / 2.0)#(TOTAL_ITEMS) #

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Label numbers:",POS_ITEMS,"positive heroes",NEG_ITEMS,"negative heroes")
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


w_array = numpy.asarray([class_weight[0], class_weight[1]])
def binary_weighted_mse(y_true, y_pred, **kwargs):
    """Implementation of mean squared error (MSE) that also accounts for class weighting

    written by jasper law
    """
    y_weights_multiplier = (keras.backend.ones_like(y_true)-y_true) * w_array[0] + y_true * w_array[1]
    sq_diff = tf.math.squared_difference(y_pred, y_true)
    rescaled = sq_diff * y_weights_multiplier
    return keras.backend.mean(rescaled, axis=-1)


def create_model(learn_rate=3.06e-5, shared_net_depth=10, final_net_depth=20, dropout_rate=0.0, weight_constraint=9999, hidden_activation="relu", output_activation="sigmoid", kernel_initializer="glorot_uniform", output_bias=INITIAL_OUTPUT_BIAS):
    """Model constructor function
    inputs ->
    shared hero subnets (+dropout) ->
    concatnate ->
    final subnet over all heroes (+dropout) ->
    output

    Model takes all heros in a single input, shape (predictors per hero, 10)
    10 Heroes are assumed
    """
    print(f"Create model: learn_rate={learn_rate}, shared_net_depth={shared_net_depth}, final_net_depth={final_net_depth}, dropout_rate={dropout_rate}, weight_constraint={weight_constraint}, hidden_activation={hidden_activation}, output_activation={output_activation}, kernel_initializer={kernel_initializer}, output_bias={output_bias}")
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    
    predictorsPerInput = variablesPerHero + nonHeroVariables
    combiInput = Input((predictorsPerInput,10))
    def dataSlice(x, heroNum):
        return x[:, heroNum]

    heroInputs = [
        Lambda(lambda x: dataSlice(x, 0), name="hero0")(combiInput),
        Lambda(lambda x: dataSlice(x, 1), name="hero1")(combiInput),
        Lambda(lambda x: dataSlice(x, 2), name="hero2")(combiInput),
        Lambda(lambda x: dataSlice(x, 3), name="hero3")(combiInput),
        Lambda(lambda x: dataSlice(x, 4), name="hero4")(combiInput),
        Lambda(lambda x: dataSlice(x, 5), name="hero5")(combiInput),
        Lambda(lambda x: dataSlice(x, 6), name="hero6")(combiInput),
        Lambda(lambda x: dataSlice(x, 7), name="hero7")(combiInput),
        Lambda(lambda x: dataSlice(x, 8), name="hero8")(combiInput),
        Lambda(lambda x: dataSlice(x, 9), name="hero9")(combiInput),
    ]
    # heroLayers = heroInputs

    # heroesDropoutInput = Dropout(dropout_rate)
    heroesSubnet = Dense(variablesPerHero + nonHeroVariables, kernel_initializer=kernel_initializer, activation=hidden_activation, name="shared1")
    heroLayers = []
    for inputLayer in heroInputs:
        heroLayers.append(heroesSubnet(inputLayer)) #Dropout(dropout_rate)

    for layerNum in range(shared_net_depth-1):
        nextLayer = Dense(variablesPerHero + nonHeroVariables, kernel_initializer=kernel_initializer, activation=hidden_activation, name="shared" + str(layerNum+2), kernel_constraint=max_norm(weight_constraint))
        # nextDropout = Dropout(dropout_rate)
        for heroNum in range(10):
            heroLayers[heroNum] = nextLayer(Dropout(dropout_rate)(heroLayers[heroNum])) # nextLayer(heroLayers[heroNum]) # nextDropout(nextLayer(heroLayers[heroNum]))

    mergeLayer = Concatenate()(heroLayers)
    if final_net_depth > 0:
        finalLayer = Dense((variablesPerHero + nonHeroVariables) * 10, kernel_initializer=kernel_initializer, activation=hidden_activation, name="final1")(mergeLayer)
        for layerNum in range(final_net_depth-1):
            finalLayer = Dense((variablesPerHero + nonHeroVariables) * 10, kernel_initializer=kernel_initializer, activation=hidden_activation, name="final" + str(layerNum + 2), kernel_constraint=max_norm(weight_constraint))(Dropout(dropout_rate)(finalLayer)) # (finalLayer)
        outputs = Dense(10, kernel_initializer=kernel_initializer, activation=output_activation, name="predictions", bias_initializer=output_bias)(finalLayer)
    else:
        outputs = Dense(10, kernel_initializer=kernel_initializer, activation=output_activation, name="predictions", bias_initializer=output_bias)(mergeLayer)

    newModel = Model(inputs=combiInput, outputs=outputs, name="TurtlePredictor")

    # Compile model
    optimizer = Adam(lr=learn_rate)
    if DO_CLASS_WEIGHTING:
        newModel.compile(loss=binary_weighted_mse, optimizer=optimizer, metrics=METRICS)
    else:
        newModel.compile(loss=keras.losses.MSE, optimizer=optimizer, metrics=METRICS)

    return newModel




##### DATA PREPROCESSING #####

# remove unused columns
dataframe.drop(["heroName" + str(heroNum) for heroNum in range(10)], axis=1, inplace=True)# + ["tick"] + ["heroTeam" + str(heroNum) for heroNum in range(10)] + ["mapControlRadiant", "mapControlDire"]

# Get header names and indices
headers = {colName: colNum for colNum, colName in enumerate(dataframe.axes[1])}

# Locate hero data
heroIDStartIndices = {}
for header, colNum in headers.items():
    if header.startswith("heroID"):
        heroIDStartIndices[int(header[len("heroID"):])] = colNum

# Calculate how many variables are used for each hero
indices = list(heroIDStartIndices.values())
# print("INDICES",indices)
variablesPerHero = indices[1] - indices[0]
nonHeroVariables = dataframe.shape[1] - 10 - (10 * variablesPerHero) 
print(variablesPerHero,"predictors per hero loaded")
print(nonHeroVariables,"non-hero predictors loaded")

##### MODEL DEFINITION
if MODEL_LOADER == "new" and ACTION == "hyperparameters":
    print("model_loader set to new, but action set to hyperparameters. skipping model creation.")
elif MODEL_LOADER == "new":
    """
    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(trainX.shape[1], input_dim=trainX.shape[1], kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    
    # evaluate model
    estimator = KerasRegressor(build_fn=baseline_model, epochs=5, batch_size=5, verbose=2)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, trainX, trainY, cv=kfold)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    """
    # with strategy.scope():
    model = create_model(learn_rate=LEARNING_RATE, shared_net_depth=SHARED_NET_DEPTH, final_net_depth=FINAL_NET_DEPTH, dropout_rate=NN_DROPOUT, weight_constraint=WEIGHT_CONSTRAIN)
    
    
    
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print("model compiled, begin training")


##### EXISTING MODEL LOADING #####

elif MODEL_LOADER == "load":
    model = keras.models.load_model(os.path.join(DATADIR, "models", MODEL_TO_LOAD))

elif MODEL_LOADER != "none":
    raise KeyError("Unrecognised model loader: " + str(MODEL_LOADER))


numTurtleTicks = -1
numNonTurtleTicks = -1


if ACTION == "plot":
    outpath = numExtFName(DATADIR + os.sep + "model_plot", ".png")
    plot_model(model, to_file=outpath, show_shapes=True, show_layer_names=True)
    print("model plot saved to " + outpath)
    exit()
elif ACTION == "data-stats":
    numTurtleTicks = sum(len(dataframe.loc[dataframe[f"isTurtling{heroNum}"] == 1]) for heroNum in range(10))
    numNonTurtleTicks = sum(len(dataframe.loc[dataframe[f"isTurtling{heroNum}"] == 0]) for heroNum in range(10))
    print("turtle ticks:",numTurtleTicks,"\nnon turtle ticks:",numNonTurtleTicks,"\ntotal:",numTurtleTicks+numNonTurtleTicks)
    exit()

# shuffle rows
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
print("rows shuffled")

dataframe["tick"] /= 360000
print("normalizing data...")
for heroNum in range(10):
    # HERO ID SCALING MOVED TO CSV READS
    # tmp = dataframe[f"heroID{heroNum}"].values.astype(tf.uint16)
    # dataframe[f"heroID{heroNum}"] = tmp / (2 ** 129)
    
    dataframe[f"posX{heroNum}"] = dataframe[f"posX{heroNum}"].subtract(-25000).div(25000-(-25000))#dataframe[f"posX{heroNum}"].subtract(8000).div(25000-8000)
    dataframe[f"posY{heroNum}"] = dataframe[f"posY{heroNum}"].subtract(-18000).div(18000-(-18000))#dataframe[f"posY{heroNum}"].subtract(16000).div(17000-16000)
    dataframe[f"posZ{heroNum}"] = dataframe[f"posZ{heroNum}"].subtract(-24000).div(24000-(-24000))#dataframe[f"posZ{heroNum}"].subtract(-1000).div(12000-(-1000))

    dataframe[f"posXPerSecond{heroNum}"] = dataframe[f"posXPerSecond{heroNum}"].subtract(-24000).div(24000-(-24000))#dataframe[f"posXPerSecond{heroNum}"].subtract(-550).div(550-(-550))
    dataframe[f"posYPerSecond{heroNum}"] = dataframe[f"posYPerSecond{heroNum}"].subtract(-18000).div(18000-(-18000))#dataframe[f"posYPerSecond{heroNum}"].subtract(-550).div(550-(-550))
    dataframe[f"posZPerSecond{heroNum}"] = dataframe[f"posZPerSecond{heroNum}"].subtract(-24000).div(24000-(-24000))#dataframe[f"posZPerSecond{heroNum}"].subtract(-550).div(550-(-550))

    dataframe[f"netWorth{heroNum}"] = dataframe[f"netWorth{heroNum}"].div(80000)
    dataframe[f"netWorthPerSecond{heroNum}"] = dataframe[f"netWorthPerSecond{heroNum}"].div(800)

    dataframe[f"kills{heroNum}"] = dataframe[f"kills{heroNum}"].div(70)
    dataframe[f"deaths{heroNum}"] = dataframe[f"deaths{heroNum}"].div(40)

    dataframe[f"XP{heroNum}"] = dataframe[f"XP{heroNum}"].clip(upper=26905)
    dataframe[f"XP{heroNum}"] = dataframe[f"XP{heroNum}"].div(26905)
    dataframe[f"XPPerSecond{heroNum}"] = dataframe[f"XPPerSecond{heroNum}"].div(40)

    dataframe[f"lastHits{heroNum}"] = dataframe[f"lastHits{heroNum}"].div(1500)
    dataframe[f"lastHitsPerSecond{heroNum}"] = dataframe[f"lastHitsPerSecond{heroNum}"].div(2.5)

    dataframe[f"closestFriendlyHeroDist{heroNum}"] = dataframe[f"closestFriendlyHeroDist{heroNum}"].div(28000)
    dataframe[f"closestFriendlyTowerDist{heroNum}"] = dataframe[f"closestFriendlyTowerDist{heroNum}"].div(28000)
    dataframe[f"closestEnemyHeroDist{heroNum}"] = dataframe[f"closestEnemyHeroDist{heroNum}"].div(28000)
    dataframe[f"closestEnemyTowerDist{heroNum}"] = dataframe[f"closestEnemyTowerDist{heroNum}"].div(28000)


print("data rescaling complete")

# convert to ndarray
dataset = dataframe.values.astype(float)

# if numpy.isnan(dataset).any():
#     raise ValueError("NAN IN DATASET")
# else:
#     raise ValueError("NO NANs")

# separate variables and targets
rawX = dataset[:,:-10]
rawY = dataset[:,-10:]


if DO_UNDERSAMPLING:
    """Under sample data by randomly dropping points
    if UNDERSAMPLING_CHANCE is -1, then number of datapoints containing at least 1 turtle will be matched by number of datapoints containing no turtles
    """
    print("doing false dropout")
    # Randomly remove some of the rows that contain no true positives
    droppedIndices = []
    falseItems = 0

    if UNDERSAMPLING_CHANCE == -1:
        trueItems = 0
        for rowNum in range(rawY.shape[0]):
            if 1 not in rawY[rowNum]:
                falseItems += 1
            else:
                trueItems += 1
        while falseItems - trueItems != 0:
            if falseItems - trueItems > 0:
                currentRow = random.randint(0,rawY.shape[0]-1)
                while currentRow in droppedIndices or 1 in rawY[currentRow]:
                    currentRow = random.randint(0,rawY.shape[0]-1)
                droppedIndices.append(currentRow)
                falseItems -= 1
            else:
                currentRow = random.randint(0,rawY.shape[0]-1)
                while currentRow in droppedIndices or 1 not in rawY[currentRow]:
                    currentRow = random.randint(0,rawY.shape[0]-1)
                droppedIndices.append(currentRow)
                trueItems -= 1
    else:
        for rowNum in range(rawY.shape[0]):
            if 1 not in rawY[rowNum]:
                if random.randint(0, 100) < UNDERSAMPLING_CHANCE:
                    droppedIndices.append(rowNum)
                falseItems += 1

    Y = numpy.ndarray(shape=(rawY.shape[0]-len(droppedIndices), rawY.shape[1]))
    X = numpy.ndarray(shape=(rawX.shape[0]-len(droppedIndices), rawX.shape[1]))
    currentRow = 0
    for rowNum in range(rawY.shape[0]):
        if rowNum not in droppedIndices:
            for cell in range(rawY.shape[1]):
                Y[currentRow, cell] = rawY[rowNum, cell]
            for cell in range(rawX.shape[1]):
                X[currentRow, cell] = rawX[rowNum, cell]
            currentRow += 1
    
    # Update class weighting
    print(falseItems,"false datapoints reduced to",falseItems - len(droppedIndices))
    print(dataset.shape[0] - falseItems,"true datapoints")
    labelCounts = [numpy.bincount(dataframe[f"isTurtling{heroNum}"]) for heroNum in range(10)]
    NEG_ITEMS, POS_ITEMS = sum([i[0] for i in labelCounts]), sum([i[1] if len(i) > 1 else 0 for i in labelCounts])
    TOTAL_ITEMS = POS_ITEMS + NEG_ITEMS
    INITIAL_OUTPUT_BIAS = numpy.log([POS_ITEMS/NEG_ITEMS])
    weight_for_0 = (1 / NEG_ITEMS) * (TOTAL_ITEMS / 2.0)
    weight_for_1 = (1 / POS_ITEMS) * (TOTAL_ITEMS / 2.0)

    class_weight[0] = weight_for_0
    class_weight[1] = weight_for_1

    print("New label numbers:",POS_ITEMS,"positive heroes",NEG_ITEMS,"negative heroes")
    print('New weight for class 0: {:.2f}'.format(weight_for_0))
    print('New weight for class 1: {:.2f}'.format(weight_for_1))
else:
    X, Y = rawX, rawY

truePoints = 0
for heroNum in range(10):
    for line in range(dataframe.shape[0]):
        if dataframe[f"isTurtling{heroNum!s}"][line] == 1:
            truePoints += 1
print(truePoints,"turtle ticks,",dataframe.shape[0]-truePoints,"datapoints with no turtles")

numTestItems = int(dataset.shape[0] * TEST_SPLIT)
numValidateItems = int(numTestItems / 2)
numTestItems -= numValidateItems

trainX = X[numTestItems+numValidateItems:]
trainY = Y[numTestItems+numValidateItems:]
testX = X[:numTestItems]
testY = Y[:numTestItems]


if ENSURE_TRAIN_POSITIVES:
    """Move datapoints with at least one turtle between train and test so train set has the percentage specified at top of this script
    """
    # Measure number of ticks containing turtles
    numTurtleTicks = len([i for i, v in enumerate(numpy.where(trainY[i] == 1)[0].size > 0 for i in range(trainY.shape[0])) if v]) + len([i for i, v in enumerate(numpy.where(testY[i] == 1)[0].size > 0 for i in range(testY.shape[0])) if v])
    # Calc number which should be in train set
    numPositivesInTrain = int(TRAIN_POSITIVE_POINTS_PERCENT * numTurtleTicks)
    # Indices of training data points that contain turtles
    sourceTurtleTickIndices = [i for i, v in enumerate(numpy.where(trainY[i] == 1)[0].size > 0 for i in range(trainY.shape[0])) if v]
    
    # Calc number which ARE in train set
    currentTrainPositives = len(sourceTurtleTickIndices)
    if currentTrainPositives == numPositivesInTrain:
        print(f"Train already contains {TRAIN_POSITIVE_POINTS_PERCENT*100}% of points with at least one positive. Doing nothing.")
    else:
        # Datapoint move direction
        moveTestToTrain = currentTrainPositives < numPositivesInTrain
        if moveTestToTrain:
            sourceTurtleTickIndices = [i for i, v in enumerate(numpy.where(testY[i] == 1)[0].size > 0 for i in range(testY.shape[0])) if v]
        # Number of points to move
        pointsToMove = abs(numPositivesInTrain - currentTrainPositives)
        print(f"Train contains {currentTrainPositives} ({(currentTrainPositives/numTurtleTicks)*100:.2f}%) of points with at least one positive, when it should have {numPositivesInTrain} {TRAIN_POSITIVE_POINTS_PERCENT*100}%. Reassigning....")
        # Sets to move from and to
        sourceX, sourceY = (testX, testY) if moveTestToTrain else (trainX, trainY)
        targetX, targetY = (trainX, trainY) if moveTestToTrain else (testX, testY)
        # indices to choose from
        targetRange = range(targetX.shape[0])
        for _ in range(pointsToMove): # for pointNum in range(pointsToMove):
            # print(pointNum,"/",pointsToMove)
            # Pick a point to move
            sourceIndex = random.choice(sourceTurtleTickIndices)
            sourceTurtleTickIndices.remove(sourceIndex)
            # Pick a point in target to swap it with
            targetIndex = random.choice(targetRange)
            # Make sure its not a turtle tick, and it isnt already marked for moving
            while 1 in targetY[targetIndex]:
                targetIndex = random.choice(targetRange)
            # swap the rows
            sourceXRow = sourceX[sourceIndex].copy()
            sourceYRow = sourceY[sourceIndex].copy()
            targetXRow = targetX[targetIndex].copy()
            targetYRow = targetY[targetIndex].copy()
            sourceX[sourceIndex] = targetXRow
            sourceY[sourceIndex] = targetYRow
            targetX[targetIndex] = sourceXRow
            targetY[targetIndex] = sourceYRow
        print(f"Positive datapoint reassigning done, {pointsToMove} datapoint pairs swapped.")
        newTrainPositives = sum(numpy.where(trainY[i] == 1)[0].size > 0 for i in range(trainY.shape[0]))
        if newTrainPositives != numPositivesInTrain:
            raise RuntimeError(f"Positive datapoint reassigning failed. Attempted to move {pointsToMove} positive points from {'test' if moveTestToTrain else 'train'} set to {'train' if moveTestToTrain else 'test'} set, but afterwards {'train' if moveTestToTrain else 'test'} set still has {newTrainPositives} ({(newTrainPositives/numTurtleTicks)*100:.2f}%) positive points. Expected to have {numPositivesInTrain} ({TRAIN_POSITIVE_POINTS_PERCENT*100}%)")

# print("trainX",trainX.shape,"trainY",trainY.shape,"testX",testX.shape,"testY",testY.shape)
print(f"trainX: ({trainX.shape[0]!s}, 10 * {(variablesPerHero + nonHeroVariables)!s} = {(10 * (variablesPerHero + nonHeroVariables))!s})\n" \
        + f"trainY: {trainY.shape}\n" \
        + f"testX: ({testX.shape[0]!s}, {(10 * (variablesPerHero + nonHeroVariables))!s})\n" \
        + f"testY: {testY.shape}")

print("shuffling columns" if DO_HERO_SHUFFLING else "arranging datasets")
# Shuffle 'columns'
# Segment the columns by hero, and randomise the order of heroes for each row
turtles = 0
nonTurtles = 0
heroData = {"hero"+str(heroNum): numpy.ndarray(shape=(trainX.shape[0], variablesPerHero + nonHeroVariables)) for heroNum in range(10)}
shuffledTrainY = numpy.ndarray(shape=trainY.shape)
for datapointNum, datapoint in enumerate(trainX):
    doneHeroes = list(str(i) for i in range(10))
    for heroID in heroIDStartIndices:
        newID = random.choice(doneHeroes) if DO_HERO_SHUFFLING else str(heroID)
        doneHeroes.remove(newID)
        if heroID == 9:
            currentData = datapoint[heroIDStartIndices[heroID]:]
        else:
            currentData = datapoint[heroIDStartIndices[heroID]:heroIDStartIndices[heroID+1]]

        heroData["hero" + newID][datapointNum, :nonHeroVariables] = trainX[datapointNum, :nonHeroVariables]
        heroData["hero" + newID][datapointNum, nonHeroVariables:] = currentData
        if trainY[datapointNum][heroID] == 1:
            turtles += 1
        else:
            nonTurtles += 1
        shuffledTrainY[datapointNum, int(newID)] = trainY[datapointNum][heroID]

# Do the same for test data
heroTestData = {"hero"+str(heroNum): numpy.ndarray(shape=(numTestItems, variablesPerHero + nonHeroVariables)) for heroNum in range(10)}
shuffledTestY = numpy.ndarray(shape=(numTestItems, 10))
heroValidateData = {"hero"+str(heroNum): numpy.ndarray(shape=(numValidateItems, variablesPerHero + nonHeroVariables)) for heroNum in range(10)}
shuffledValidateY = numpy.ndarray(shape=(numValidateItems, 10))

for datapointNum, datapoint in enumerate(testX):
    doneHeroes = list(str(i) for i in range(10))
    for heroID in heroIDStartIndices:
        newID = random.choice(doneHeroes) if DO_HERO_SHUFFLING else str(heroID)
        doneHeroes.remove(newID)
        if heroID == 9:
            currentData = datapoint[heroIDStartIndices[heroID]:]
        else:
            currentData = datapoint[heroIDStartIndices[heroID]:heroIDStartIndices[heroID+1]]

        if datapointNum >= numTestItems:
            heroValidateData["hero" + newID][datapointNum-numTestItems, :nonHeroVariables] = testX[datapointNum, :nonHeroVariables]
            heroValidateData["hero" + newID][datapointNum-numTestItems, nonHeroVariables:] = currentData
            shuffledValidateY[datapointNum-numTestItems, int(newID)] = testY[datapointNum][heroID]
        else:
            heroTestData["hero" + newID][datapointNum, :nonHeroVariables] = testX[datapointNum, :nonHeroVariables]
            heroTestData["hero" + newID][datapointNum, nonHeroVariables:] = currentData
            shuffledTestY[datapointNum, int(newID)] = testY[datapointNum][heroID]

        if testY[datapointNum][heroID] == 1:
            turtles += 1
        else:
            nonTurtles += 1





stackedTestingData = numpy.stack(list(heroTestData.values()), axis=-1)
stackedTrainingData = numpy.stack(list(heroData.values()), axis=-1)
stackedValidationData = numpy.stack(list(heroValidateData.values()), axis=-1)

if USE_TF_DATASET:
    testData_asDataset = tf.data.Dataset.from_tensor_slices((numpy.asarray([numpy.asarray([stackedTestingData[line, :, :]]) for line in range(stackedTestingData.shape[0])]), shuffledTestY))
    trainData_asDataset = tf.data.Dataset.from_tensor_slices((numpy.asarray([numpy.asarray([stackedTrainingData[line, :, :]]) for line in range(stackedTrainingData.shape[0])]), shuffledTrainY))
    validateData_asDataset = tf.data.Dataset.from_tensor_slices((numpy.asarray([numpy.asarray([stackedValidationData[line, :, :]]) for line in range(stackedValidationData.shape[0])]), shuffledValidateY))

print("column shuffling complete" if DO_HERO_SHUFFLING else "data arrangement complete")
print(turtles,"turtling hero tick segments",nonTurtles,"non-turtling hero tick segments")

print("preprocessing complete, begin building model")


if ACTION == "hyperparameters":
    """Perform linear search over any number hyperparams to find the local optimum
    """
    # raise ValueError("STACKEDTESTINGDATA: " + str(stackedTestingData.shape) + "\nstackedValidationData: " + str(stackedValidationData.shape))
    # raise ValueError("stackedTrainingData: " + str(stackedTrainingData.shape) + "\nshuffledTrainY: " + str(shuffledTrainY.shape) + "\nstackedTestingData: " + str(stackedTestingData.shape) + "\nshuffledTestY: " + str(shuffledTestY.shape))

    # staticParams = dict(epochs=64, batch_size=128, learn_rate=1e-03, shared_net_depth=16, final_net_depth=16, dropout_rate=0, weight_constraint=5)
    # paramSetsToTest = [
    #                 dict(searchParamName="epochs", searchParamValues=[32, 64, 128]),
    #                 dict(searchParamName="batch_size", searchParamValues=[32, 64, 128]),
    #                 dict(searchParamName="learn_rate", searchParamValues=[0.00001, 0.001, 0.01, 0.2]),
    #                 dict(searchParamName="shared_net_depth", searchParamValues=[16, 64, 128]),
    #                 dict(searchParamName="final_net_depth", searchParamValues=[16, 64, 128]),
    #                 dict(searchParamName="dropout_rate", searchParamValues=[0, 0.3, 0.6, 0.9]),
    #                 dict(searchParamName="weight_constraint", searchParamValues=[1, 3, 5])]

    # staticParams = dict(epochs=16, batch_size=64, learn_rate=1e-05, shared_net_depth=30, final_net_depth=16, dropout_rate=0.1, weight_constraint=2)
    # paramSetsToTest = [
    #                 dict(searchParamName="epochs", searchParamValues=[16, 64, 128]),
    #                 dict(searchParamName="batch_size", searchParamValues=[64, 128]),
    #                 dict(searchParamName="learn_rate", searchParamValues=[0.000001, 0.00001, 0.00005]),
    #                 dict(searchParamName="shared_net_depth", searchParamValues=[8, 16, 30]),
    #                 dict(searchParamName="final_net_depth", searchParamValues=[4, 16, 50, 80])]
    #                 # dict(searchParamName="dropout_rate", searchParamValues=[0, 0.3, 0.6, 0.9]),
    #                 # dict(searchParamName="weight_constraint", searchParamValues=[1, 3, 5])]

    # staticParams = dict(epochs=16, batch_size=64, learn_rate=1e-06, shared_net_depth=26, final_net_depth=4, dropout_rate=0.1, weight_constraint=2)
    # paramSetsToTest = [
    #                 dict(searchParamName="epochs", searchParamValues=[16, 64, 128]),
    #                 dict(searchParamName="batch_size", searchParamValues=[32, 64]),
    #                 dict(searchParamName="learn_rate", searchParamValues=[1e-7, 5e-7, 1e-6]),
    #                 dict(searchParamName="shared_net_depth", searchParamValues=[20, 23, 26, 29]),
    #                 dict(searchParamName="final_net_depth", searchParamValues=[1, 2, 3, 4])]
    #                 # dict(searchParamName="dropout_rate", searchParamValues=[0, 0.3, 0.6, 0.9]),
    #                 # dict(searchParamName="weight_constraint", searchParamValues=[1, 3, 5])]

    # staticParams=dict(epochs=64, batch_size=32, learn_rate=1e-06, shared_net_depth=29, final_net_depth=3, dropout_rate=0.1, weight_constraint=2, output_activation="sigmoid")
    # paramSetsToTest = [dict(searchParamName="output_activation", searchParamValues=["sigmoid", "softmax"])]


    # staticParams = dict(epochs=16, batch_size=32, learn_rate=1e-06, shared_net_depth=10, final_net_depth=16, dropout_rate=0.1, weight_constraint=2)
    # paramSetsToTest = [
    #                 dict(searchParamName="shared_net_depth", searchParamValues=[3,6,9,12,15]),
    #                 dict(searchParamName="final_net_depth", searchParamValues=[3,6,9,12,15]),
    #                 dict(searchParamName="batch_size", searchParamValues=[16, 32, 64, 128]),
    #                 dict(searchParamName="epochs", searchParamValues=[4,8,16,32]),
    #                 # dict(searchParamName="learn_rate", searchParamValues=[0.00001, 0.001, 0.01, 0.2]),
    #                 # dict(searchParamName="dropout_rate", searchParamValues=[0, 0.3, 0.6, 0.9]),
    #                 # dict(searchParamName="weight_constraint", searchParamValues=[1, 3, 5])]
    #                 ]


    # StaticParams is the "current local optimum" - param values used when this hyperparam is currently being explored
    staticParams = dict(epochs=4, batch_size=32, learn_rate=1e-05, shared_net_depth=8, final_net_depth=8, dropout_rate=0.0, weight_constraint=5)
    # Each hyperparam is tested in turn, but searches for each hyperparam are parallelized where possible.
    paramSetsToTest = [
                    dict(searchParamName="epochs", searchParamValues=[4, 8, 16, 32]),
                    dict(searchParamName="batch_size", searchParamValues=[16, 32, 64, 128]),
                    dict(searchParamName="learn_rate", searchParamValues=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3]),
                    dict(searchParamName="shared_net_depth", searchParamValues=[4, 8, 16, 32]),
                    dict(searchParamName="final_net_depth", searchParamValues=[4, 8, 16, 32]),
                    dict(searchParamName="dropout_rate", searchParamValues=[0, 0.3, 0.6, 0.9]),
                    dict(searchParamName="weight_constraint", searchParamValues=[1, 3, 5])]

    for testParams in paramSetsToTest:
        if testParams["searchParamName"] not in staticParams:
            raise ValueError("param not assigned static: " + str(testParams["searchParamName"]))
        if staticParams[testParams["searchParamName"]] not in testParams["searchParamValues"]:
            print("##!## Adding missing static param value to searched values: " + str(testParams["searchParamName"]) + "=" + str(staticParams[testParams["searchParamName"]]))
            testParams["searchParamValues"].append(staticParams[testParams["searchParamName"]])

    # staticParams = staticParams=dict(epochs=1, batch_size=64, learn_rate=1e-05, shared_net_depth=1, final_net_depth=1, dropout_rate=0.0, weight_constraint=2)
    # paramSetsToTest = [
    #                 dict(searchParamName="weight_constraint", searchParamValues=[1])]

    results = {}
    def testParams(currentParams):
        if not currentParams["searchParamValues"]:
            print("##!##! ERR: No values provided for param '" + currentParams["searchParamName"] + "', skipping test")
        else:
            if NUM_PARALLEL_GPUS == 1:
                for v in currentParams["searchParamValues"]:
                    results[currentParams["searchParamName"]] = {}
                    if USE_TF_DATASET:
                        results[currentParams["searchParamName"]].update(singleHyperParamSearch_inputAsDataset(create_model, staticParams, currentParams["searchParamName"], [v], trainData_asDataset, testData_asDataset, validateData_asDataset, (stackedTrainingData.shape, shuffledTrainY.shape), (stackedTestingData.shape, shuffledTestY.shape), (stackedValidationData.shape, shuffledValidateY.shape), currentParams["gpusToUse"]))
                    else:
                        results[currentParams["searchParamName"]].update(singleHyperParamSearch(create_model, staticParams, currentParams["searchParamName"], [v], stackedTrainingData, shuffledTrainY, stackedTestingData, shuffledTestY, stackedValidationData, shuffledValidateY, currentParams["gpusToUse"]))
            else:
                if USE_TF_DATASET:
                    results[currentParams["searchParamName"]].update(singleHyperParamSearch_inputAsDataset(create_model, staticParams, currentParams["searchParamName"], currentParams["searchParamValues"], trainData_asDataset, testData_asDataset, validateData_asDataset, (stackedTrainingData.shape, shuffledTrainY.shape), (stackedTestingData.shape, shuffledTestY.shape), (stackedValidationData.shape, shuffledValidateY.shape), currentParams["gpusToUse"]))
                else:
                    results[currentParams["searchParamName"]] = singleHyperParamSearch(create_model, staticParams, currentParams["searchParamName"], currentParams["searchParamValues"], stackedTrainingData, shuffledTrainY, stackedTestingData, shuffledTestY, stackedValidationData, shuffledValidateY, currentParams["gpusToUse"])
            print("#!# " + currentParams["searchParamName"] + " test complete:\n - " + "\n - ".join(str(k) + ": "  + str(v) for k, v in results[currentParams["searchParamName"]].items()))

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for testNum, paramsToTest in enumerate(paramSetsToTest):
            print(f"##!## Beginning hyperparam search test subset {testNum+1}/{len(paramSetsToTest)}...")
            results.clear()
            if USE_GPU:
                gpusToUse = pickLeastUsedGPUs(numGPUs=min(len(paramsToTest["searchParamValues"]), NUM_PARALLEL_GPUS)) if NUM_PARALLEL_GPUS != 1 else [0]
            else:
                gpusToRemove = [0]
            paramsToTest["gpusToUse"] = gpusToUse
            testParams(paramsToTest)

            optimalValues = {}
            for k in results:
                if len(results[k]) == 0:
                    print(f"###!### ERR: No results found for test '{k}' - was its param set empty?")
                else:
                    testedVals = list(results[k].keys())
                    bestKey = testedVals[0]
                    if len(results[k]) > 1:
                        for k1 in testedVals[1:]:
                            if results[k][k1] < results[k][bestKey]:
                                bestKey = k1
                    optimalValues[k] = bestKey
            
            if optimalValues:
                print("##!## Search test subset complete. staticParams updated with the following locally optimal values:\n  - " + \
                        "\n  - ".join((k + "=" + str(v) + " (" + str(results[k][v]) + ")") for k, v in optimalValues.items()))
                
                results["static_params"] = staticParams

                outpath = numExtFName(DATADIR + os.sep + f"search-results/hyperparam search part {testNum}", ".json")

                with open(outpath, "w") as f:
                    json.dump(results, f, indent=4, sort_keys=True)

                print("##!## Search test subset results saved to: " + outpath)
                staticParams.update(optimalValues)
            else:
                print("##!## Search test subset complete. No findings or changes made.")

        # results = singleHyperParamSearch(create_model, {}, "shared_net_depth", [4,16,32], stackedTestingData, shuffledTestY, stackedValidationData, shuffledValidateY)
        # print(" - " + "\n - ".join(str(k) + ": "  + str(v) for k, v in results.items()))


elif ACTION == "evaluate":
    """Train a model on train set, evaluate on test set.
    If EVALUATE, also save predictions for test set,
    """
    # PARALLELIZED FITTING
    if USE_TF_DATASET:
        history = model.fit(trainData_asDataset, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2, shuffle=True, callbacks=[early_stopping], validation_data=validateData_asDataset)#, class_weight=class_weight)
    # SERIALIZED FITTING
    else:
        history = model.fit(x=stackedTrainingData, y=shuffledTrainY, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2, shuffle=True)#, callbacks=[early_stopping], validation_data=(stackedValidationData, shuffledValidateY))#, class_weight=class_weight)

    # PARALLELIZED EVALUATION
    if USE_TF_DATASET:
        results = model.evaluate(testData_asDataset)

    # SERIALIZED EVALUATION
    else:
        results = model.evaluate(x=stackedTestingData, y=shuffledTestY)

    # print("losses:", ", ".join("%.2f" % result for result in results))
    print("training complete. Losses:", ", ".join("%.4f" % result for result in results))
    
    ##### MODEL EVALUATION #####

    if EVALUATE:
        print("begin evaluation")
        heroColumns = nonHeroVariables + (variablesPerHero) * 10
        numTests = int(POST_EVAL_PERCENT * testX.shape[0])
        predictionsShape = (numTests, heroColumns + testY.shape[1] + 10)
        predictionsData = numpy.zeros(shape=predictionsShape)
        predictionsData[:, :nonHeroVariables] = heroTestData["hero0"][:numTests, :nonHeroVariables]
        # for colName, colNum in headers.items():
        #     predictionsData[0, colNum] = colName

        # for colNum in range(10):
        #     predictionsData[0, -(10-colNum)] = "predictedIsTurtling" + str(colNum)

        predictionsData[:,:heroIDStartIndices[0]] = heroTestData["hero0"][:numTests, :nonHeroVariables]
        for hero in heroTestData:
            heroNum = int(hero[len("hero"):])
            predictionsData[:,heroIDStartIndices[heroNum]:heroIDStartIndices[heroNum] + variablesPerHero] = heroTestData[hero][:numTests, nonHeroVariables:]
        
        # predictionsData[: predictionsShape[0], :testX.shape[1]] = heroTestData
        predictionsData[:,heroColumns:heroColumns + testY.shape[1]] = shuffledTestY[:numTests]

        if USE_GPU:
            with tf.device("/GPU:" + str(pickLeastUsedGPUs(numGPUs=1)[0])):
                # PARALLELIZED TESTING
                if USE_TF_DATASET:
                    predictionsData[:,-10:] = model.predict(testData_asDataset)
                else:
                # SERIALIZED TESTING
                    for testNum in range(numTests):
                        if testNum % 100 == 0:
                            print(str(int((testNum / numTests) * 100)) + "% Predicting test " + str(testNum) + "/" + str(numTests-1))
                        predictionsData[testNum, heroColumns + testY.shape[1]:] = model.predict(numpy.asarray([stackedTestingData[testNum, :, :]]))
        else:
            # PARALLELIZED TESTING
            if USE_TF_DATASET:
                predictionsData[:,-10:] = model.predict(testData_asDataset)
            else:
            # SERIALIZED TESTING
                for testNum in range(numTests):
                    if testNum % 100 == 0:
                        print(str(int((testNum / numTests) * 100)) + "% Predicting test " + str(testNum) + "/" + str(numTests-1))
                    predictionsData[testNum, heroColumns + testY.shape[1]:] = model.predict(numpy.asarray([stackedTestingData[testNum, :, :]]))


    ##### MODEL EVALUATION SAVING #####
    if SAVE_EVALUATION:
        outpath = numExtFName(os.path.join(DATADIR, sys.argv[1].split(os.sep)[-1].split(".")[0] + "-predictions"), ".csv")
        
        DataFrame(predictionsData, columns=dataHeaders).to_csv(outpath, index=False)
        # numpy.savetxt(outpath, predictionsData, delimiter=",")
        print("evaluation complete, results saved to",outpath)
    if SAVE_MODEL:
        print("saving model...")

        modelDir = os.path.join(DATADIR, "models", datetime.now().strftime("%m-%d-%Y_%H-%M"))
        model.save(modelDir)
        totalDatapoints = X.shape[0]
        paramNames = ("MODEL_LOADER", "USE_TF_DATASET", "TEST_SPLIT", "LEARNING_RATE", "SHARED_NET_DEPTH", "FINAL_NET_DEPTH", "BATCH_SIZE", "FALSE_DROPOUT_CHANCE",
                        "TEST_SPLIT", "DO_FALSE_DROPOUT", "DO_HERO_SHUFFLING", "NN_DROPOUT", "WEIGHT_CONSTRAIN", "NUM_EPOCHS",
                        "turtleHeroes", "nonTurtleHeroes", "variablesPerHero", "nonHeroVariables", "turtles", "nonTurtles", "totalDatapoints")
        depth1Params = ("trainX.shape", "trainY.shape", "shuffledTestY.shape", "shuffledValidateY.shape")
        thisModule = sys.modules[__name__]

        with open(os.path.join(modelDir, "TRAINING PARAMS.toml"), "w") as f:
            f.write("\n".join(vName + " = " + str(getattr(thisModule, vName)) for vName in paramNames) + "\n" + "\n".join("[" + v.split(".")[0] + "]\n" + v.split(".")[1] + " = " + str(getattr(getattr(thisModule, v.split(".")[0]), v.split(".")[1])) for v in depth1Params))

        print("model and hyperparameters saved to " + modelDir)

    plot_metrics(history)
    if not USE_GPU:
        plt.show()
    else:
        plt.savefig(os.path.join(DATADIR, datetime.now().strftime("evaluate-%m-%d-%Y_%H-%M.png")))
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()

elif ACTION == "plot_weights":
    """Plot the updating of weights throughout training. Really useful to see if the model is actually learning or not."""
    weights_history = []
    class MyCallback(keras.callbacks.Callback):
        """https://www.moxleystratton.com/tensorflow-visualizing-weights/"""
        def on_batch_end(self, batch, logs):
            weights_history.append([w[0] for w in model.get_weights()[0]])
    
    callback = MyCallback()

    # PARALLELIZED FITTING
    if USE_TF_DATASET:
        model.fit(trainData_asDataset, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2, shuffle=True, callbacks=[callback, early_stopping])#, class_weight=class_weight)
    # SERIALIZED FITTING
    else:
        model.fit(x=stackedTrainingData, y=shuffledTrainY, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2, shuffle=True, callbacks=[callback, early_stopping])#, class_weight=class_weight)

    # PARALLELIZED EVALUATION
    if USE_TF_DATASET:
        results = model.evaluate(testData_asDataset)

    # SERIALIZED EVALUATION
    else:
        results = model.evaluate(x=stackedTestingData, y=shuffledTestY)

    # print("losses:", ", ".join("%.2f" % result for result in results))
    print("training complete. Loss: %.4f" % results)

    plt.figure(1, figsize=(6, 3))
    plt.plot(weights_history)
    plt.ylabel("Neuron weights")
    plt.xlabel("Training batch")
    if not USE_GPU:
        plt.show()
    else:
        plt.savefig(os.path.join(DATADIR, datetime.now().strftime("plot_weights-%m-%d-%Y_%H-%M.png")))
