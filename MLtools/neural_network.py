#!/bin/env python

"""
Neural network utilities

S.D. Butalla & M. Rahmani
2021/06/07
"""
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.initializers import RandomNormal
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow.keras import backend as keras

# scikit-optimize packages
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.plots import plot_objective
from skopt.plots import plot_evaluations
from skopt.plots import plot_histogram
from skopt.plots import plot_objective_2D
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})


# import sys, os
# sys.path.append('/Users/stephenbutalla1/Desktop/FD_Model_ML/utilities/')
# from matchingUtilities import *
# #from ..utilities.matchingUtilities import *

# def preprocessData(fileName, permBool = False, diMu_dRBoolean = False):
#     '''
#     This function is currently not working! Need to fix
#     Opens file and preprocesses data
#     '''
#     selMu_eta, selMu_phi, selMu_pt, selMu_charge, genAMu_eta, genAMu_phi, genAMu_pt, genAMu_charge = extractData(fileName)

#     badEvents = prelimCuts(selMu_phi, selMu_eta, selMu_pt, selMu_charge, genAMu_eta, verbose = True)

#     selMu_etaCut = removeBadEvents(badEvents, selMu_eta)
#     selMu_phiCut = removeBadEvents(badEvents, selMu_phi)
#     selMu_ptCut = removeBadEvents(badEvents, selMu_pt)
#     selMu_chargeCut = removeBadEvents(badEvents, selMu_charge)
#     genAMu_etaCut = removeBadEvents(badEvents, genAMu_eta)
#     genAMu_phiCut = removeBadEvents(badEvents, genAMu_phi)
#     genAMu_ptCut = removeBadEvents(badEvents, genAMu_pt)
#     genAMu_chargeCut = removeBadEvents(badEvents, genAMu_charge)

#     dRgenFinal = dRgenCalc(genAMu_etaCut, selMu_etaCut, genAMu_phiCut, selMu_phiCut)

#     min_dRgen, dRgenMatched = SSM(dRgenFinal, genAMu_chargeCut, selMu_chargeCut, verbose = False, extraInfo = False)

#     wrongPerm = Wpermutation(min_dRgen, selMu_chargeCut, genAMu_chargeCut, twoPerm = True)

#     invariantMass = invMassCalc(min_dRgen, wrongPerm, selMu_ptCut, selMu_etaCut, selMu_phiCut)

#     allPerm = allPerms(min_dRgen, wrongPerm)

#     diMu_dR = dR_diMu(min_dRgen, wrongPerm, selMu_phiCut, selMu_etaCut)

#     finalDataShaped = fillFinalArray(selMu_ptCut, selMu_etaCut, selMu_phiCut, selMu_chargeCut, allPerm, invariantMass, diMu_dR, perm = permBool, diMu_dRBool = diMu_dRBoolean, pandas = False)

#     return finalDataShaped


def prepareData(dataframe, testSize, scalerType=None, returnDL=False):
    """
    ====== INPUTS ======
    dataframe: final data array with features and labels (labels in last column)
    testSize: size of the test set, expressed as a decimal
    scaler:    Type of data preprocessing method
        Default: None
        Options: "StandardScaler", "MaxAbsScaler", "Normalizer", "MinMaxScaler", "PowerTransformer", "QuantileTransformer", "RobustScaler"
    returnDL:  Boolean variable. If True, return just the (optionally scaled) data and labels in separate arrays
    """

    # Separate data from labels
    data = dataframe[:, 0 : (dataframe.shape[1] - 1)]
    labels = dataframe[:, (dataframe.shape[1] - 1)].astype(int)

    if scalerType is None:
        pass
    elif scalerType == "StandardScaler":
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
    elif scalerType == "MaxAbsScaler":
        scaler = MaxAbsScaler().fit(data)
        data = scaler.transform(data)
    elif scalerType == "MaxAbsScaler":
        scaler = MinMaxScaler().fit(data)
        data = scaler.transform(data)
    elif scalerType == "Normalizer":
        scaler = Normalizer().fit(data)
        data = scaler.transform(data)
    elif scalerType == "PowerTransformer":
        scaler = PowerTransformer().fit(data)
        data = scaler.transform(data)
    elif scalerType == "QuantileTransformer":
        scaler = QuantileTransformer().fit(data)
        data = scaler.transform(data)
    elif scalerType == "RobustScaler":
        scaler = RobustScaler().fit(data)
        data = scaler.transform(data)

    if returnDL == True:
        return data, labels
    else:
        pass

    trainX, testX, trainY, testY = train_test_split(
        data, labels, test_size=testSize, random_state=1
    )  # Set seed to 1 for reproducibility

    return trainX, testX, trainY, testY


# def getZDMasses(dirName, cutBelow = 125):
#     '''
#     Given base directory for all ROOT files, returns
#     all ZD masses
#     '''
#     subDirs = [folder.path for folder in os.scandir(dirName) if folder.is_dir()]
#     mZDmasses = []

#     for directory in subDirs:
#         tempStr = directory.split('/')
#         temp = tempStr[6].split('_')
#         mZDmasses.append(temp[1])

#     mZDmasses = np.array(mZDmasses, dtype = int)
#     mZDmasses.sort() # sort in ascending order
#     mZDmasses = np.unique(mZDmasses)
#     mask      = np.where(mZDmasses < cutBelow)[0]

#     return np.delete(mZDmasses, mask)

# def getfDMasses(mZDmass, cutBelow = 0):
#     mZDdirName = 'MZD_' + str(mZDmass)
#     files = glob.glob('ROOT_Files/' + mZDdirName + '/*.root')

#     fDmasses = []
#     for file in files:
#         temp = file.split('/')
#         temp = temp[2].split('_')
#         temp = temp[2].split('.')
#         fDmasses.append(temp[0])

#     fDmasses = np.array(fDmasses, dtype = int)
#     fDmasses.sort() # sort in ascending order
#     fDmasses = np.unique(fDmasses)
#     mask     = np.where(fDmasses < cutBelow)[0]

#     return np.delete(fDmasses, mask)

# def runNet(trainX, trainY, model, maxIt, batchSize, callback = True):
#     '''
#     Runs neural network with a 20% validation split (if input size of training data is 70% of original dataset).
#     Returns history for use with the plotting function.
#     '''
#     earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1) # Stop training if validation loss doesn't improve
#     saveModel = ModelCheckpoint('neuralNetworkResults/curentbest.hdf5', save_best_only=True, monitor='val_loss', mode='min') # Save model where best results acheived
#     redLR     = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min') # Reduce learning rate when loss plateaus

#     history = model.fit(trainX, trainY, validation_split = 0.285, epochs = maxIt, callbacks = [earlyStop, saveModel, redLR], verbose = 1)
#     # batch_size = batchSize,
#     return history

# def netResults(testX, testY, rootFileName, dirName):
#     '''
#     Prints classification report to standard out
#     and saves results to directory
#     '''
#     masses  = splitFileName(rootFileName)
#     fileName = 'results_mZD-' + masses[0] + '_mfD-' + masses[1] + '.txt'
#     file = open(dirName + '/' + fileName, "w")
#     print('Test accuracy')
#     predictedY = model.predict(testX)
#     predictedY = predictedY.argmax(axis=1)
#     print('======== Neural network results ========')
#     print('mZD: {} mfD: {}\n'.format(masses[0], masses[1]))
#     print(classification_report(testY, predictedY))
#     print('Overall test accuracy')

#     loss, accuracy = model.evaluate(testX, testY, verbose=1)
#     print('Test loss: {}'.format(loss))
#     print('Test accuracy: {}'.format(accuracy))
#     file.write(str(accuracy))
#     file.close()

#     return accuracy

# def plotNetResults(history, plotName, maxIt, dirName):
#     '''
#     Plots accuracy and loss as a function of epoch
#     and saves to a file in the directory created.
#     '''
#     fig, ax1 = plt.subplots(figsize=(12,8))
#     ax1.plot(np.arange(0, maxIt), history.history["accuracy"], color='tab:red', label="Training accuracy")
#     ax1.plot(np.arange(0, maxIt), history.history["val_accuracy"], color='tab:blue', label="Validation accuracy")
#     ax1.set_ylim(0,1)
#     #ax.set_title("Training Loss and Accuracy")
#     ax1.set_xlabel("Epoch", loc='right')
#     ax1.set_ylabel("Accuracy", loc='top')
#     ax1.grid()
#     ax1.legend()
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Loss', loc='top')  # we already handled the x-label with ax1

#     maxY = max(history.history["val_loss"])
#     ax2.plot(np.arange(0, maxIt), history.history["loss"], color='tab:green', label="Training loss")
#     ax2.plot(np.arange(0, maxIt), history.history["val_loss"], color='tab:orange', label="Validation loss")
#     ax2.set_ylim(0, maxY + 50)
#     ax2.legend(loc='upper left')

#     name = dirName + '/' + plotName + '.pdf'
#     plt.savefig(name)
#     plt.show()

#     return None


# def splitFileName(fileName):
#     tempName = fileName.split('.')
#     masses = tempName[0].split('_')
#     m_ZD = masses[1]
#     m_fD = masses[2]

#     return [m_ZD, m_fD]
