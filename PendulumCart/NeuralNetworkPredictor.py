"""
This file provides implementations of the neural network
"""

import numpy as np 
import PlantModel.InvertedPendulum as ip 
import tensorflow as tf 
from sklearn import utils as skutil
import copy 
from matplotlib import pyplot as plt
from DatasetGenerator import DatasetGenerator


class SecondOrderNNPredictorWithStateInput():
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

    def __init__(self, features, labels, hiddenLayerSize = 20, 
                        trainingEpoch = 2500, trainingSetPortion = 0.8,
                        finiteDifferenceDu = 0.001):
        """
        set up training set, test set and the structure of the network
        """
        self.features = copy.deepcopy(features)
        self.labels = copy.deepcopy(labels)
        self.hiddenLayerSize = hiddenLayerSize 
        self.trainingEpoch = trainingEpoch
        if (hiddenLayerSize > 100) or (trainingEpoch > 6000):
            print("WARNING: too many hidden units or training epochs. May have overfitting")

        self.finiteDifferenceDu = finiteDifferenceDu
        skutil.shuffle(self.features, self.labels)
        self.datasetLength = int(self.features.shape[0])
        self.trainingSetLength = int(self.datasetLength * trainingSetPortion)
        self.testSetLength = int(self.datasetLength - self.trainingSetLength)
        self.trainingFeature = self.features[0:self.trainingSetLength, :]
        self.trainingLabel = self.labels[0:self.trainingSetLength, :]
        self.testFeature = self.features[self.trainingSetLength:, :]
        self.testLabel = self.labels[self.trainingSetLength:, :]


        self.inputLayer = tf.keras.layers.Input(batch_input_shape = [self.trainingSetLength, 3],
                                    name = 'input')
        self.hiddenLayer = tf.keras.layers.Dense(hiddenLayerSize, activation = 'sigmoid')
        self.hiddenLayerOutput = self.hiddenLayer(self.inputLayer)

        self.outputLayer = tf.keras.layers.Dense(1, activation = None)
        self.finalOutput = self.outputLayer(self.hiddenLayerOutput)

        self.model = tf.keras.Model(inputs = [self.inputLayer], outputs = [self.finalOutput]) 
        self.model.compile(optimizer = tf.keras.optimizers.Adam(0.01), 
                            loss = tf.keras.losses.MeanSquaredError())

        self.model.summary()
        self.model.fit(self.trainingFeature, self.trainingLabel, batch_size = self.trainingSetLength,
                                    epochs = self.trainingEpoch)
        print("Training completed. evaluating on test set.")
        self.model.evaluate(self.testFeature, self.testLabel)

        self.hiddenLayerWeight = self.hiddenLayer.weights[0].numpy()
        self.hiddenLayerBias = self.hiddenLayer.weights[1].numpy()
        self.outputLayerWeight = self.outputLayer.weights[0].numpy()
        self.outputLayerBias = self.outputLayer.weights[1].numpy()

    
    def plotPredictions(self, inputX1, inputX2):
        """
        try to plot the result of NN predictions
        to show the estimation of output and derivatives intuitively
        """
        evaluateFeature = []
        evaluateLabel = []
        for curData in range(self.datasetLength):
            if self.features[curData][0] == inputX1 and self.features[curData][1] == inputX2:
                evaluateFeature.append([self.features[curData][0], self.features[curData][1],
                                        self.features[curData][2]])
                evaluateLabel.append([self.labels[curData][0]])
        
        evaluateFeature = np.array(evaluateFeature)
        evaluateLabel = np.array(evaluateLabel)
        
        
        evaluateLength = evaluateFeature.shape[0]

        prediction = self.model(evaluateFeature)
        prediction = prediction.numpy()
        prediction = prediction.reshape((evaluateLength,))

        plt.plot(prediction,'x-', label = 'Neural network output')
        evaluateLabel = evaluateLabel.reshape((evaluateLength,))
        plt.plot(evaluateLabel, 's-', label = 'Expected output')
        plt.legend()
        plt.show()
    
    def getLookaheadPredictionWithTensorflow(self, systemInput, systemStateX1, systemStateX2):
        """
        DEPRECRATED
        Get the estimation of system output after a fixed period T
        Here the output is calculated using tensorflow.
        """
        nnInput = np.array([[systemStateX1, systemStateX2, systemInput]]) 
        prediction = self.model(nnInput)
        prediction = prediction.numpy()
        print(prediction) 
        return prediction[0][0]
    
    def getLookaheadDerivativePredictionWithTensorflow(self, systemInput, systemStateX1, systemStateX2):
        """
        DEPRECATED
        Get the estimation of the system output derivative with respect to u
        Note that this is still based on tensorflow, using finite difference method.
        """
        nnInput1 = np.array([[systemStateX1, systemStateX2, systemInput]])
        nnInput2 = np.array([[systemStateX1, systemStateX2, systemInput + self.finiteDifferenceDu]])
        prediction1 = self.model(nnInput1).numpy()[0][0]
        prediction2 = self.model(nnInput2).numpy()[0][0]
        
        print((prediction2 - prediction1)/self.finiteDifferenceDu)
        return (prediction2 - prediction1)/self.finiteDifferenceDu
    
    def getLookaheadPredictionAnalytical(self, systemInput, systemStateX1, systemStateX2):
        """
        Calculate the predicted output after  time T using analytical solution
        Not calling tensorflow API, will be faster.
        """
        nnInput = np.array([[systemStateX1, systemStateX2, systemInput]])
        hiddenOutput = self.sigmoid(np.matmul(nnInput, self.hiddenLayerWeight) + self.hiddenLayerBias)
        finalOutput = np.matmul(hiddenOutput, self.outputLayerWeight) + self.outputLayerBias
        return finalOutput[0][0]

    def getLookaheadDerivativePredictionAnalytical(self, systemInput, systemStateX1, systemStateX2):
        """
        Calculate the derivative using analytical solution
        """
        nnInput = np.array([[systemStateX1, systemStateX2, systemInput]])
        w3 = np.array([self.hiddenLayerWeight[2,:]])
        dy1du = self.sigmoidDerivative(np.matmul(nnInput, self.hiddenLayerWeight) + self.hiddenLayerBias) * w3 
        dy2dy1 = self.outputLayerWeight
        dy2du = np.matmul(dy1du, dy2dy1)

        return dy2du[0][0]



def inputSignal(t):
		return -np.pi / 6 + np.pi * np.sin(t) / 3*0.8

if __name__ == '__main__':
    """
    Test tracking using model-free NR controller
    """
    np.random.seed(2021)
    tf.random.set_seed(2022)
    lookaheadTime = 0.2
    datasetGenerator = DatasetGenerator()
    invertedPendulum = ip.InvertedPendulum(lookaheadTime = lookaheadTime)
    datasetGenerator.setPlantDynamics(invertedPendulum)
    feature, label = datasetGenerator.getSecondOrderStateInput()
   

    secondOrderNNPredictorWithStateInput = SecondOrderNNPredictorWithStateInput(feature, label)
   
    # test the neural network using inverted pendulum

    simulationDt = 0.01 
    simulationTime = 25
    simulationStep = int(simulationTime / simulationDt) 
    controllerSpeed = 35
    
    MSE = 0
    inputList = [0]
    referenceSignal = [inputSignal(0)]
    for curStep in range(1, simulationStep):
        predictionX1 = secondOrderNNPredictorWithStateInput.getLookaheadPredictionAnalytical(inputList[-1],
                                    invertedPendulum.stateX1, invertedPendulum.stateX2)

        derivative = secondOrderNNPredictorWithStateInput.getLookaheadDerivativePredictionAnalytical(inputList[-1],
                                    invertedPendulum.stateX1, invertedPendulum.stateX2)
        
        predictionSignal = inputSignal(curStep * simulationDt + lookaheadTime)
        referenceSignal.append(inputSignal(curStep * simulationDt))
        invertedPendulum.runSystem([inputList[-1]])

        du = controllerSpeed * (predictionSignal - predictionX1) / derivative 
        
        
        inputList.append(inputList[-1] + du * simulationDt)


    plt.plot(invertedPendulum.stateX1History,  label = 'System output')
    plt.plot(referenceSignal, '--', label = 'Reference signal')
    absError = np.abs(np.array(invertedPendulum.stateX1History) - np.array(referenceSignal))
    #plt.plot(absError, label = 'Error in absolute value')
    
    mseError = np.sum(absError ** 2)/absError.shape[0]
    plt.legend()
    plt.xlabel("Sample index (0.01s)")
    plt.ylabel("Outputs (rad)")
    plt.grid()
    plt.title("Tracking result using model-free method, inverted pendulum\n Lookahead time = %.2fs, controller speed = %.2f, MSE = %.4f"%(lookaheadTime, controllerSpeed, mseError))
    plt.figure()
    plt.grid()
    plt.xlabel("Sample index (0.01s)")
    plt.ylabel("u(t)")
    plt.title("u(t) using model-free method, inverted pendulum\n Lookahead time = %.2f, controller speed = %.2f"%(lookaheadTime, controllerSpeed))
    plt.plot(inputList, label = "System input u(t)")
    plt.legend()

    plt.show()
