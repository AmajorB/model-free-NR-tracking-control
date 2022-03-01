"""
This script trains the neural network and save the model in a given directory
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
import os
import math 



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

    def __init__(self, features = [], labels = [], featureDimension = 5, labelDimension = 2, hiddenLayerSize = 35,
                        hiddenLayer2Size = 200, 
                        trainingEpoch = 5, trainingSetPortion = 1,
                        finiteDifferenceDu = 0.001, plantDynamics = None,
                        loadSavedModel = False, continueTraining = False, modelName =  None,
                        modelLayers = 1, trainingRound = 1):
        """
        set up training set, test set and the structure of the network
        """
    
        self.features = copy.deepcopy(features)
        self.featureDimension = featureDimension 
        self.labels = copy.deepcopy(labels)

        self.labelDimension = labelDimension
        self.hiddenLayerSize = hiddenLayerSize 
        self.plantDynamics = plantDynamics
        self.trainingEpoch = trainingEpoch
        self.modelLayers = modelLayers

        self.finiteDifferenceDu = finiteDifferenceDu
        
        if not loadSavedModel:    
            self.inputLayer = tf.keras.layers.Input(shape = (self.featureDimension,),
                                        name = 'input')
                                        
            self.hiddenLayer = tf.keras.layers.Dense(hiddenLayerSize, activation = 'sigmoid')
            self.hiddenLayerOutput = self.hiddenLayer(self.inputLayer)

            self.outputLayer = tf.keras.layers.Dense(self.labelDimension, activation = None)

            self.finalOutput = self.outputLayer(self.hiddenLayerOutput)

            self.model = tf.keras.Model(inputs = [self.inputLayer], outputs = [self.finalOutput]) 
            
            self.model.compile(optimizer = tf.keras.optimizers.Adam(0.01), 
                                loss = tf.keras.losses.MeanSquaredError())
            self.model.summary()
            time.sleep(5)

            self.features, self.labels = skutil.shuffle(self.features, self.labels)
            self.datasetLength = int(self.features.shape[0])
            self.trainingSetLength = int(self.datasetLength * trainingSetPortion)
            self.testSetLength = int(self.datasetLength - self.trainingSetLength)
            self.trainingFeature = self.features[0:self.trainingSetLength, :]
            self.trainingLabel = self.labels[0:self.trainingSetLength, :]
            self.testFeature = self.features[self.trainingSetLength:, :]
            self.testLabel = self.labels[self.trainingSetLength:, :]
            print(self.trainingSetLength, " training data")
           
            for curRound in range(trainingRound):
                self.model.fit(self.trainingFeature,  self.trainingLabel, 
                            batch_size = self.trainingSetLength, epochs = self.trainingEpoch)

                print("Training completed. evaluating on test set.")
                if trainingSetPortion == 1:
                    evaluateResult = 0
                else:
                    evaluateResult = self.model.evaluate(self.testFeature, self.testLabel, batch_size = self.testSetLength)
                
                evaluateResultTrainingSet = self.model.evaluate(self.trainingFeature, self.trainingLabel, batch_size = self.trainingSetLength)
                if os.name == 'posix':
                    self.model.save((".\\ModelSave\\MultipleTrain\\Single50T0.5P1Enlarged\\" + time.strftime("%b-%d-%H-%M", time.localtime()) + ',%.6f,%.6f'%(evaluateResult, evaluateResultTrainingSet)).replace('\\','/'))
                else:
                    self.model.save(".\\ModelSave\\MultipleTrain\\Single35T0.5P1Enlarged1.5pi\\" + time.strftime("%b-%d-%H-%M", time.localtime()) + ',%.6f,%.6f'%(evaluateResult, evaluateResultTrainingSet))
            self.hiddenLayerWeight = self.hiddenLayer.weights[0].numpy()
            self.hiddenLayerBias = self.hiddenLayer.weights[1].numpy()
            self.outputLayerWeight = self.outputLayer.weights[0].numpy()
            self.outputLayerBias = self.outputLayer.weights[1].numpy()
        else:
            self.model = tf.keras.models.load_model('.\\ModelSave\\'+modelName)

            
        
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
        """
        try to plot the result of NN predictions
        to show the estimation of output and derivatives intuitively
        """
        evaluateInput = np.array([[0.5, 0.5, 0.1, 0, np.pi/4, 0, 0, 0]])
        evaluateU = [0.1 * i for i in range(-20,20,1)]
        evaluateOutput = []
        referenceOutput = []
        for curU in evaluateU:
            evaluateInput[0][5] = curU 
            curOutput = self.model(evaluateInput).numpy()
            evaluateOutput.append(curOutput[0][0])
            systemStates = evaluateInput.tolist()[0][0:6]
            systemInputs = evaluateInput.tolist()[0][6:8]
            curRef = self.plantDynamics.predictSystem(systemInputs, systemStates)
            referenceOutput.append(curRef[0])
        print(evaluateOutput)
        plt.plot(evaluateU, evaluateOutput, 'o', label = 'nn output')
        plt.plot(evaluateU, referenceOutput, 's', label = 'reference output')
        plt.legend()
        plt.show()
   
if __name__ == '__main__':
    lookaheadTime = 0.5
    
    modelLayers = 1
    loadFeatureName = '.\\DataSave\\featureT0.5,8,8,pi+0.5pi,8,8Enlarged.npy'
    loadLabelName = '.\\DataSave\\labelT0.5,8,8,pi+0.5pi,8,8Elnarged.npy'
    if os.name == 'posix':
        loadFeatureName = loadFeatureName.replace('\\', '/')
        loadLabelName = loadLabelName.replace('\\', '/')
    trainingEpoch = 1000
    trainingRound = 200

    np.random.seed(8888)
    tf.random.set_seed(8888)
    
    
    feature = np.load(loadFeatureName)
    label = np.load(loadLabelName)

    nnPredictorWithStateInput = NNPredictorWithStateInput(feature, label, 
            plantDynamics= dc.DubinsCar(lookaheadTime = lookaheadTime), loadSavedModel= False, modelLayers= modelLayers, 
            trainingEpoch= trainingEpoch, trainingRound= trainingRound)
    
