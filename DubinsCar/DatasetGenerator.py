"""
DatasetGenerator samples data from the target dynamical system
and return data that can be directly feed into the neural network.
"""

import numpy as np
from scipy.sparse import data 
import DubinsCar as dc
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

    def getThirdOrderStateInput(self, x1Min = -8, x1Max = 8, x1Cnt =6,
                                x2Min = -8, x2Max = 8, x2Cnt = 6, 
                                x3Min = -np.pi, x3Max = np.pi, x3Cnt = 6,
                                u1Min = -8, u1Max = 8, u1Cnt = 4,
                                u2Min = -8, u2Max = 8, u2Cnt = 4):
    
        if (self.plantDynamics == None):
            print("Please first set the plant dynamics!")
            exit(-1)

        x1Sample = np.linspace(x1Min, x1Max, x1Cnt + 1)
        x2Sample = np.linspace(x2Min, x2Max, x2Cnt + 1)
        x3Sample = np.linspace(x3Min, x3Max, x3Cnt + 1)
        u1Sample = np.linspace(u1Min, u1Max, u1Cnt + 1)
        u2Sample = np.linspace(u2Min, u2Max, u2Cnt + 1)
        x1Length = len(x1Sample)
        x2Length = len(x2Sample)
        x3Length = len(x3Sample)
        u1Length = len(u1Sample)
        u2Length = len(u2Sample)
        print('Dataset sample points: %d %d %d %d %d'%(x1Length, x2Length, x3Length,
                                                                u1Length, u2Length))
        self.datasetLength = x1Length * x2Length * x3Length * u1Length * u2Length
        print('%d pieces of data created!'%self.datasetLength)
        self.featureData = np.zeros([self.datasetLength, 5])
        self.labelData = np.zeros([self.datasetLength, 2])
        dataCount = 0
       
        for curX1 in x1Sample:
            for curX2 in x2Sample:
                for curX3 in x3Sample:
                    for curU1 in u1Sample:
                        for curU2 in u2Sample:
                            self.featureData[dataCount][0] = curX1 
                            self.featureData[dataCount][1] = curX2 
                            self.featureData[dataCount][2] = curX3
                            self.featureData[dataCount][3] = curU1 
                            self.featureData[dataCount][4] = curU2 

                            x1Prediction, x2Prediction, x3Prediction = self.plantDynamics.predictSystem(curU1, curU2,
                            stateX1 = curX1, stateX2 = curX2, stateX3 = curX3)
                            
                            self.labelData[dataCount][0] = x1Prediction
                            self.labelData[dataCount][1] = x2Prediction     
                            dataCount += 1 
            
        return self.featureData, self.labelData

if __name__ == '__main__':
    lookaheadTime = 0.5
    datasetGenerator = DatasetGenerator()
    dubinsCar = dc.DubinsCar(lookaheadTime = lookaheadTime)
    datasetGenerator.setPlantDynamics(dubinsCar)
    feature, label = datasetGenerator.getThirdOrderStateInput()
    print(feature.shape)
    np.save('./DataSave/featureT0.5,8,8,pi,8,8Enlarged.npy', feature)
    np.save('./DataSave/labelT0.5,8,8,pi,8,8Elnarged.npy', label)
    print(feature[0:5])
    print(label[0:5])