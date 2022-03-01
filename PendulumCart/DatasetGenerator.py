"""
DatasetGenerator samples data from the target dynamical system
and return data that can be directly feed into the neural network.
"""

import numpy as np 
import PlantModel.InvertedPendulum as ip 
class DatasetGenerator:
    def __init__(self):
        self.datasetLength = None 
        self.featureData = None 
        self.labelData = None  
        self.plantDynamics = None 
    
    def setPlantDynamics(self, newPlant):
        """
        Set the dynamics of the plant you wish to sample
        Note that this function should be called before calling the next one (create dataest)
        Otherwise, no plant will be sampled.
        """
        self.plantDynamics = newPlant 

    def getSecondOrderStateInput(self, x1Min = -2, x1Max = 2, x1Cnt =5,
                                x2Min = -2, x2Max = 2, x2Cnt = 5, 
                                uMin = -80, uMax = 80, uCnt = 10):
        """
        Param x1Min, x1Max: set the range of the first state
        Param x2Min, x2Max: set the range of the second state 
        Param uMin, uMax: set the range of the input
        Param x1Cnt, x2Cnt, uCnt: define how many data will be sampled in the corresponding range
        """
        if (self.plantDynamics == None):
            print("Please first set the plant dynamics!")
            exit(-1)

        x1Sample = np.linspace(x1Min, x1Max, x1Cnt + 1)
        x2Sample = np.linspace(x2Min, x2Max, x2Cnt + 1)
        uSample = np.linspace(uMin, uMax, uCnt + 1)
        x1Length = len(x1Sample)
        x2Length = len(x2Sample)
        uLength = len(uSample)
        print('Dataset sample points: %d %d %d'%(x1Length, x2Length, uLength))
        self.datasetLength = x1Length * x2Length * uLength
        print('%d pieces of data created!'%self.datasetLength)
        self.featureData = np.zeros([self.datasetLength, 3])
        self.labelData = np.zeros([self.datasetLength, 1])
        dataCount = 0
        """
        featureData[i][0]: x1
        featureData[i][1]: x2
        featureData[i][2]: u
        """
        for curX1 in x1Sample:
            for curX2 in x2Sample:
                for curU in uSample:
                    self.featureData[dataCount][0] = curX1 
                    self.featureData[dataCount][1] = curX2 
                    self.featureData[dataCount][2] = curU 

                    x1Prediction, x2Prediction = self.plantDynamics.predictSystem(curU, stateX1 = curX1, stateX2 = curX2)
                    
                    self.labelData[dataCount][0] = x1Prediction     
                    dataCount += 1 
        
        return self.featureData, self.labelData
    
     