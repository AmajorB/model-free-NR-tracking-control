"""
This file provides the dubins car model, or the "differentially driven" car model.
This is different from the traditional dubins car, because it has two inputs.
"""

import numpy as np 
from matplotlib import pyplot as plt 
import math 

class DubinsCar():
    def __init__(self, initialX1 = 0, initialX2 = 0, initialX3 = 0, simulationDt = 0.01, lookaheadTime = 0.5, 
                    odeStepSize = 0.01, systemParamR = 0.05, systemParamD = 0.20):
        """
        To define a plant, we need its states and the simulation stepsize.
        variable step stands for how many steps of simulation has finished.
        """
        self.stateX1 = initialX1 
        self.stateX2 = initialX2 
        self.stateX3 = initialX3 
        self.simulationDt = simulationDt
        self.lookaheadTime =lookaheadTime
        self.odeStepSize = odeStepSize
        self.systemParamR = systemParamR
        self.systemParamD = systemParamD 

        self.stateX1History = [self.stateX1]
        self.stateX2History = [self.stateX2]
        self.stateX3History = [self.stateX3]

    def setSystemState(self, stateX1, stateX2, stateX3):
        """
        Set the state of the system
        """
        self.stateX1 = stateX1
        self.stateX2 = stateX2
        self.stateX3 = stateX3 

    
    def getSystemState(self, stateX1 = None, stateX2 = None, stateX3 = None):
        if (stateX1 == None) or (stateX2 == None) or (stateX3 == None):
            return self.stateX1, self.stateX2, self.stateX3 
        else:
            return stateX1, stateX2, stateX3 
    
    def getSystemNextState(self, systemInput1, systemInput2, stateX1 = None, stateX2 = None, stateX3 = None ):
        """
        Obtain the next state according to current state and current input.
        No change to the states should be made in this function.
        """
        if (stateX1 == None) or (stateX2 == None) or (stateX3 == None):
            stateX1 = self.stateX1
            stateX2 = self.stateX2
            stateX3 = self.stateX3 

        dx1 = self.systemParamR * np.cos(stateX3) * (systemInput1 + systemInput2)
        dx2 = self.systemParamR * np.sin(stateX3) * (systemInput1 + systemInput2)
        dx3 = self.systemParamR/self.systemParamD*(systemInput1 - systemInput2)

        nextX1 = stateX1 + dx1 * self.simulationDt
        nextX2 = stateX2 + dx2 * self.simulationDt
        nextX3 = stateX3 + dx3 * self.simulationDt
        
        return nextX1, nextX2, nextX3 

    def getSystemNextOutput(self, systemInput1, systemInput2, stateX1 = None, stateX2 = None, stateX3 = None ):
        """
        From current state and current input, obtain the next system output.
        """
        nextX1, nextX2, nextX3 = self.getSystemNextState(systemInput1, systemInput2, stateX1, stateX2, stateX3)
        return nextX1, nextX2

    def runSystem(self, systemInputList, biasX = 0, biasY = 0):
        """
        Run the system with a given list of inputs
        This function changes the state of the system.
        """		
        for curInput in systemInputList:
            self.stateX1, self.stateX2, self.stateX3 = self.getSystemNextState(curInput[0], curInput[1])
            originalX1 = self.stateX1 + biasX
            originalX2 = self.stateX2 + biasY 
            self.stateX1History.append(originalX1)
            self.stateX2History.append(originalX2) 
            self.stateX3History.append(self.stateX3)
        return self.stateX1History, self.stateX2History, self.stateX3History 
    
    def predictSystem(self, systemInput1, systemInput2, stateX1 = None, stateX2 = None, stateX3 = None):
        """
        Run a lookahead simulation to determine the output of the system
        after a lookahead time T.
        This function does not change the state of the system.
        """
        lookaheadStep = math.floor(self.lookaheadTime/self.simulationDt)
        if (stateX1 == None) or (stateX2 == None) or (stateX3 == None):
            stateX1 = self.stateX1 
            stateX2 = self.stateX2
            stateX3 = self.stateX3 

        for curStep in range(lookaheadStep):
            stateX1, stateX2, stateX3 = self.getSystemNextState(systemInput1,
                                systemInput2, stateX1, stateX2, stateX3)
        
        return stateX1, stateX2, stateX3 
    
    def getPredictionDerivative(self, systemInput1, systemInput2, stateX1 = None, stateX2 = None, stateX3 = None):
        """
        Calculate the derivative of the predictions with respect to input u
        Using Forward-Euler method (legacy)
        This function does not change the state of the model.
        """
        if (stateX1 == None) or (stateX2 == None) or (stateX3 == None):
            stateX1 = self.stateX1
            stateX2 = self.stateX2 
            stateX3 = self.stateX3
        odeStep = math.floor(self.lookaheadTime/self.odeStepSize)
        curDxDu = np.zeros([3,2])

    
        matA = np.array([[0, 0, -self.systemParamR*np.sin(stateX3)*(systemInput1 + systemInput2)],
                        [0, 0,   self.systemParamR*np.cos(stateX3)*(systemInput1 + systemInput2)],
                        [0, 0, 0]])

        b = np.array([[self.systemParamR * np.cos(stateX3), self.systemParamR * np.cos(stateX3)], 
                    [self.systemParamR * np.sin(stateX3), self.systemParamR * np.sin(stateX3)],
                    [self.systemParamR/self.systemParamD, -self.systemParamR/self.systemParamD]])
        for curStep in range(odeStep):
            dxduDt = matA @ curDxDu + b
            curDxDu += dxduDt * self.odeStepSize
        dhDx = np.array([[1, 0, 0], [0, 1, 0]])
        dgDu = dhDx @ curDxDu
        
        return  dgDu
    
    def getPredictionDerivativeFiniteDifference(self, systemInput1, systemInput2, 
    stateX1 = None, stateX2 = None, stateX3 = None ):
        """
        Calculate the derivative of the predictions with respect to input u
        Using Forward-Euler method (legacy)
        This function does not change the state of the model.
        """
        if (stateX1 == None) or (stateX2 == None) or (stateX3 == None):
            stateX1 = self.stateX1
            stateX2 = self.stateX2 
            stateX3 = self.stateX3 
            
        predictX11, _ = self.predictSystem(systemInput1, systemInput2, stateX1, stateX2, stateX3)
        predictX12, _ = self.predictSystem(systemInput1 + 0.001, systemInput2, stateX1, stateX2)
        
        print('prediction=', predictX11, predictX12)
        return  (predictX12 - predictX11)/0.001
