"""
This script test a trained neural network model using the differentially driven car
"""

import numpy as np 
import DubinsCar as dc 
import tensorflow as tf 
tf.config.experimental.set_visible_devices([], 'GPU')
from sklearn import utils as skutil
import copy 
from matplotlib import pyplot as plt
from DatasetGenerator import DatasetGenerator
import time 
import math 
import os 


class NNPredictorWithStateInput():
    def sigmoid(self, x):
        """
        Sigmoid (logistic) function
        """
        return 1/(1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        """
        Derivative of sigmoid (logistic) function
        """
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def __init__(self, modelName, modelLayers, plantDynamics = None):
        """
        set up training set, test set and the structure of the network
        """
        self.modelLayers = modelLayers

        if os.name == 'posix':
            self.model = tf.keras.models.load_model(modelName.replace('\\','/'))
        else:
            self.model = tf.keras.models.load_model(modelName)
           
        
        if modelLayers == 2:
            self.layer1 = self.model.layers[1]
            self.layer2 = self.model.layers[2]
            self.layerOut = self.model.layers[3]
            
            self.layer1Weight = self.layer1.weights[0].numpy()
            self.layer1Bias = self.layer1.weights[1].numpy()

            self.layer2Weight = self.layer2.weights[0].numpy()
            self.layer2Bias = self.layer2.weights[1].numpy()

            self.finalWeight = self.layerOut.weights[0].numpy()
            self.finalBias = self.layerOut.weights[1].numpy()
        else:
            self.layer1 = self.model.layers[1]
            self.layerOut = self.model.layers[2]
            self.layer1Weight = self.layer1.weights[0].numpy()
            self.layer1Bias = self.layer1.weights[1].numpy()

            self.finalWeight = self.layerOut.weights[0].numpy()
            self.finalBias = self.layerOut.weights[1].numpy()


    def plotPredictions(self):
        pass 

    def getLookaheadPredictionWithTensorflow(self, systemInput1, systemInput2, 
                                systemStateX1, systemStateX2, systemStateX3):
        nnInput = np.array([[systemStateX1, systemStateX2, systemStateX3, systemInput1, systemInput2]])
        
        prediction = self.model(nnInput)

        prediction = prediction.numpy()
        return prediction
    
    def getLookaheadDerivativePredictionAnalytical(self, systemInput1, systemInput2, 
                                systemStateX1, systemStateX2, systemStateX3):

        nnInput = np.array([[systemStateX1, systemStateX2, systemStateX3, systemInput1, systemInput2]])
        
        if self.modelLayers == 2:

            
            dy1dx = self.sigmoidDerivative(np.matmul(nnInput, self.layer1Weight) + self.layer1Bias).T * self.layer1Weight.T
            y1 = self.sigmoid(nnInput @ self.layer1Weight + self.layer1Bias)
            dy2dy1 = self.sigmoidDerivative(np.matmul(y1, self.layer2Weight) + self.layer2Bias).T * self.layer2Weight.T
            y2 = self.sigmoid(y1 @ self.layer2Weight + self.layer2Bias)
            dy3dy2 = self.finalWeight.T
            dy3dx = dy3dy2 @ dy2dy1 @ dy1dx 
            return dy3dx[0:2,3:5]

        else:
            
            dy1dx = self.sigmoidDerivative(np.matmul(nnInput, self.layer1Weight) + self.layer1Bias).T * self.layer1Weight.T
            y1 = self.sigmoid(nnInput @ self.layer1Weight + self.layer1Bias)
            
            dy3dy2 = self.finalWeight.T
            dy3dx = dy3dy2 @ dy1dx 
 
            return dy3dx[0:2,3:5]

def refSignal(t):
    # reference signal with transitional signals
    if t<=5:
        return np.array([[-0.0001*(t**3) + 0.25*t],[0.0475*(t**3) - 0.3601*(t**2) + 0.3*t + 3]])

    return np.array([[5*np.sin(0.05*t)], [3*np.sin(0.1*t)]])

def plotResult(x1History, x2History, refHistory, uHistory, recenterList, showPlot = False, mode = None,
            transientSamplePoint = 700):
    # show the simulation results
    if mode == "Traditional":
        simulationType = "model-based"
    else:
        simulationType = "model-free"
    refX = []
    refY = []
    errorX = []
    errorY = []
    totalMSE = 0
    maxError = 0
    maxErrorAfterTransient = 0
    maxErrorAfterTransientPos = 0
    inputU1 = []
    inputU2 = []
    length = min(len(x1History),len(x2History))
    X = x1History[0:length]
    Y = x2History[0:length]
    length = min(length, len(refHistory))

    for curRef in range(length):
        refX.append(refHistory[curRef][0])
        refY.append(refHistory[curRef][1])
        errorX.append(refX[curRef] - X[curRef])
        errorY.append(refY[curRef] - Y[curRef])
        inputU1.append(uHistory[curRef][0])
        inputU2.append(uHistory[curRef][1])
        curSE = errorX[-1]**2 + errorY[-1]**2
        maxError = max(maxError, curSE**0.5)
        if curRef > transientSamplePoint:
            if maxErrorAfterTransient < curSE**0.5:
                maxErrorAfterTransient = curSE**0.5 
                maxErrorAfterTransientPos = curRef 
        totalMSE += curSE 
    print("max error = %.6f, MSE = %.6f, max error after transient:%.6f at location %d"%(maxError, totalMSE/length, maxErrorAfterTransient,
        maxErrorAfterTransientPos))
    if showPlot:
        plt.plot(x1History, x2History, linewidth = 2, label = 'System output')
        plt.plot(refX, refY,'--', linewidth = 2, label = 'Reference signal')
        plt.title("Tracking result using " +simulationType +  " method\n Differentially driven car\nLookaheadTime = 0.50s, controller speed = 20")
        plt.xlabel("X1-position (m)")
        plt.ylabel("X2-position (m)")
        plt.grid()
        plt.legend()
        plt.figure()

        plt.title("Tracking result of X1-position using "+ simulationType+ " method\n Differentially driven car")
        plt.plot(X, linewidth = 2, label = 'System output X1-position')
        plt.plot(refX, '--', linewidth = 2, label = 'Reference signal X1-position')
    
        
        plt.xlabel('Sample index (0.01s)')
        plt.ylabel('X1-position (m)')
        plt.grid()

        plt.legend()
        plt.figure()
        plt.title("Tracking result of X2-position using " + simulationType + " method\n Differentially driven car")
        plt.plot(Y, linewidth = 2, label = 'System iutput X2-position')
        plt.plot(refY, '--', linewidth = 2, label = 'Reference signal X2-position')
        plt.xlabel("Sample index (0.01s)")
        plt.ylabel("X2-position (m)")
        plt.grid()
        plt.legend()
        plt.figure()


        plt.plot(inputU1, linewidth = 2, label = 'Right wheel')
        plt.plot(inputU2, linewidth = 2, label = 'Left wheel')
        plt.grid()
        plt.xlabel("Sampling index (0.01s)")
        plt.ylabel("v$_r$(t), v$_l$(t) (rad/s)")
        plt.title("Input control u(t), " + simulationType + " method\n Differentially driven car")
        for curRecenter in recenterList:
            plt.plot([curRecenter, curRecenter],[-5, 5], 'r-')
        plt.legend()
        plt.figure()
        plt.title('Tracking error in x1, '+ simulationType + " method")
        plt.plot(errorX, label = "X1 error")
        plt.xlabel("Sampling index /0.01s")
        plt.ylabel("Error in x1 /m")
        plt.legend()
        plt.grid()
        plt.figure()
        plt.title('Tracking error in x2, ' + simulationType + " method")
        plt.plot(errorY, label = "X2 error")
        plt.legend()
        plt.xlabel("Sampling index /0.01s")
        plt.ylabel("Error in x2 /m")
        plt.grid()

        plt.figure()
        refX1 = refX[0: 500]
        refY1 = refY[0: 500]
        refX2 = refX[500:]
        refY2 = refY[500:]
        plt.plot(refX1, refY1, 'r-', label = "Reference signal, modified part")
        plt.plot(refX2, refY2, 'b-', label = "Reference signal, original part")
        plt.legend(loc ="lower center")
        plt.title("Reference signal for differentially driven car")
        plt.xlabel("X1-position")
        plt.ylabel("X2-position")
        plt.grid()

        plt.show()
    return max(errorX), max(errorY)

        


if __name__ == '__main__':
    
    MODE = 'NN' # run model free Newton-Raphson tracking control
    #MODE = "Traditional" # run model based Newton-Raphson tracking control
    useSaturation = False 
    initialX1 = 0 
    initialX2 = 3
    
    initialX3 = 0.876058
    simulationDt = 0.01 
    simulationTime = 3.141592653589*40 + 5
    simulationStep = int(simulationTime / simulationDt) 
    controllerSpeed = 20
    lookaheadTime = 0.5
    
    modelLayers = 1
    
    
    modelDirectory = '.\\ModelSave\\MultipleTrain\\Single35T0.5P1Enlarged1.5pi\\'
    singleModelName = 'Oct-05-12-14,0.000000,0.000154'

    loadModelName = modelDirectory + singleModelName

    neuralNetworkPredictor = NNPredictorWithStateInput(loadModelName, modelLayers)

    dubinsCar = dc.DubinsCar(initialX1, initialX2, initialX3, lookaheadTime= lookaheadTime, simulationDt = simulationDt)
    
    inputList = [[1, 0]]
    referenceSignal = [refSignal(0)]
    biasX = 0
    biasY = 0
    recenterList = []
    hasNegativeInput = False 
    earlyTermination = False 
    for curStep in range(0, simulationStep):

        if MODE == 'Traditional':
            predictionX1, predictionX2, predictionX3 = dubinsCar.predictSystem(inputList[-1][0], inputList[-1][1])
            predictionX = np.array([[predictionX1],[predictionX2]])
            derivative = dubinsCar.getPredictionDerivative(inputList[-1][0], inputList[-1][1])

        if MODE == 'NN':
            predictionX = neuralNetworkPredictor.getLookaheadPredictionWithTensorflow(inputList[-1][0], inputList[-1][1],
                                            dubinsCar.stateX1, dubinsCar.stateX2, dubinsCar.stateX3)
            
            predictionX = predictionX.reshape([2,1])
            derivative = neuralNetworkPredictor.getLookaheadDerivativePredictionAnalytical(inputList[-1][0], inputList[-1][1],
                                            dubinsCar.stateX1, dubinsCar.stateX2, dubinsCar.stateX3)
            
                                            
        predictionSignal = refSignal(curStep * simulationDt + lookaheadTime) - np.array([[biasX],[biasY]])
        dubinsCar.runSystem([inputList[-1]], biasX, biasY)
        try:
            du = controllerSpeed * np.linalg.inv(derivative)@(predictionSignal - predictionX) 
        except np.linalg.LinAlgError:
            earlyTermination = True 
            break 
        nextInput1 = inputList[-1][0] + du[0][0] * simulationDt
        nextInput2 = inputList[-1][1] + du[1][0] * simulationDt
        
        inputList.append([nextInput1, nextInput2])
        
        referenceSignal.append(refSignal(curStep * simulationDt))


    errorX, errorY = plotResult(dubinsCar.stateX1History, dubinsCar.stateX2History, referenceSignal,
            inputList, recenterList, showPlot= not earlyTermination, mode = MODE )
    print('model '+ curModel +' Xerror = %.4f Yerror = %.4f'%(errorX, errorY) + str(earlyTermination) + str(hasNegativeInput))
       