"""
This file provides a simple model for inverted pendulum
"""

import numpy as np 
from matplotlib import pyplot as plt 
import math 

class InvertedPendulum():
    def __init__(self, initialX1 = np.pi/6, initialX2 = 0, simulationDt = 0.01, lookaheadTime = 0.15, 
                    odeStepSize = 0.001, systemParamBigM = 1, systemParamSmallM = 0.2, 
                        systemParamL = 2):
        """
        To define a plant, we need its states and the simulation stepsize.
        variable step stands for how many steps of simulation has finished.
        """
        self.stateX1 = initialX1 
        self.stateX2 = initialX2 
        self.simulationDt = simulationDt
        self.lookaheadTime =lookaheadTime
        self.odeStepSize = odeStepSize
        self.systemParamBigM = systemParamBigM
        self.systemParamSmallM = systemParamSmallM 
        self.systemParamL = systemParamL 
        self.gravitationAcc = 9.81

        self.stateX1History = [self.stateX1]
        self.stateX2History = [self.stateX2]

    def setSystemState(self, stateX1, stateX2):
        """
        Set the state of the system
        """
        self.stateX1 = stateX1
        self.stateX2 = stateX2

    def getSystemOutput(self, stateX1 = None, stateX2 = None):
        """
        From system state to system output.
        y = x1
        """
        if (stateX1 == None) or (stateX2 == None):
            return self.stateX1
        else:
            return stateX1 
    
    def getSystemState(self, stateX1 = None, stateX2 = None):
        """
        Get current state of the system
        """
        if (stateX1 == None) or (stateX2 == None):
            return self.stateX1, self.stateX2 
        else:
            return stateX1, stateX2 
    
    def getSystemNextState(self, systemInput, stateX1 = None, stateX2 = None):
        """
        Obtain the next state according to current state and current input.
        No change to the states should be made in this function.
        """
        if (stateX1 == None) or (stateX2 == None):
            stateX1 = self.stateX1
            stateX2 = self.stateX2

        dx1 = stateX2 
        dx2 = -(self.systemParamSmallM * self.systemParamL * (stateX2**2) * np.sin(stateX1) * np.cos(stateX1) \
                - (self.systemParamSmallM + self.systemParamBigM) * self.gravitationAcc * np.sin(stateX1)  
                        -np.cos(stateX1) * systemInput)/  \
                    (self.systemParamBigM * self.systemParamL + self.systemParamSmallM * self.systemParamL * (np.sin(stateX1)**2))
        nextX1 = stateX1 + dx1 * self.simulationDt
        nextX2 = stateX2 + dx2 * self.simulationDt
        return nextX1, nextX2 

    def getSystemNextOutput(self, systemInput, stateX1 = None, stateX2 = None):
        """
        From current state and current input, obtain the next system output.
        """
        nextX1, nextX2 = self.getSystemNextState(systemInput, stateX1, stateX2)
        return nextX1 

    def runSystem(self, systemInputList):
        """
        Run the system with a given list of inputs
        This function changes the state of the system.
        """		
        for curInput in systemInputList:
            self.stateX1, self.stateX2 = self.getSystemNextState(curInput)
            self.stateX1History.append(self.stateX1)
            self.stateX2History.append(self.stateX2) 
        return self.stateX1History, self.stateX2History 
    
    def predictSystem(self, systemInput, stateX1 = None, stateX2 = None):
        """
        Run a lookahead simulation to determine the output of the system
        after a lookahead time T.
        This function does not change the state of the system.
        """
        lookaheadStep = math.floor(self.lookaheadTime/self.simulationDt)
        if (stateX1 == None) or (stateX2 == None):
            stateX1 = self.stateX1 
            stateX2 = self.stateX2

        for curStep in range(lookaheadStep):
            dx1 = stateX2 
            dx2 = -(self.systemParamSmallM * self.systemParamL * (stateX2**2) * np.sin(stateX1) * np.cos(stateX1) \
                    - (self.systemParamSmallM + self.systemParamBigM) * self.gravitationAcc * np.sin(stateX1) 
                       -     np.cos(stateX1) * systemInput)/  \
                        (self.systemParamBigM * self.systemParamL + self.systemParamSmallM * self.systemParamL * (np.sin(stateX1)**2))
            stateX1 += dx1 * self.simulationDt
            stateX2 += dx2 * self.simulationDt 
        
        return stateX1, stateX2 
    
    def getPredictionDerivative(self, systemInput, stateX1 = None, stateX2 = None):
        """
        Calculate the derivative of the predictions with respect to input u
        Using Forward-Euler method (legacy)
        This function does not change the state of the model.
        """
        if (stateX1 == None) or (stateX2 == None):
            stateX1 = self.stateX1
            stateX2 = self.stateX2 
        odeStep = math.floor(self.lookaheadTime/self.odeStepSize)
        curDxDu = np.zeros([2,1])
        M = self.systemParamBigM
        m = self.systemParamSmallM
        l = self.systemParamL
        x1 = stateX1 
        x2 = stateX2 
    
        g = self.gravitationAcc

        matA = np.array([[0, 1], 
        [((-m*l*(x2**2)*np.cos(2*x1) + (m + M)*g*np.cos(x1) - systemInput * np.sin(x1)) * (M * l + m * l * np.sin(x1)**2) - 
                (-m*l*(x2**2)*np.sin(x1)*np.cos(x1) + (m + M)*g*np.sin(x1) + systemInput * np.cos(x1))*(2*m*l*np.sin(x1)*np.cos(x1)))/
                ((M*l + m*l*(np.sin(x1)**2))**2), 
                -2*m*l*(x2**2)*np.sin(x1) * np.cos(x1)/(M*l + m*l * (np.sin(x1)**2))]])
        b = np.array([[0], [np.cos(x1)/(M*l + m*l * (np.sin(x1)**2))]])
        for curStep in range(odeStep):
            dxduDt = matA @ curDxDu + b
            curDxDu += dxduDt * self.odeStepSize
        dhDx = np.array([[1, 0]])
        dgDu = dhDx @ curDxDu
        
        return  dgDu[0][0]
    
    def getPredictionDerivativeFiniteDifference(self, systemInput, stateX1 = None, stateX2 = None):
        """
        Calculate the derivative of the predictions with respect to input u
        Using Forward-Euler method (legacy)
        This function does not change the state of the model.
        """
        if (stateX1 == None) or (stateX2 == None):
            stateX1 = self.stateX1
            stateX2 = self.stateX2 
            
        predictX11, _ = self.predictSystem(systemInput, stateX1, stateX2)
        predictX12, _ = self.predictSystem(systemInput + 0.1, stateX1, stateX2)
        print('prediction=', predictX11, predictX12)
        return  (predictX12 - predictX11)/0.1

def inputSignal(t):
		return -np.pi / 6 + np.pi * np.sin(t) / 3*0.8


if __name__ == '__main__':
    """
    Test tracking using model-based Newton-Raphson controller
    """
    initialX1 = np.pi/6
    initialX2 = 0
    simulationDt = 0.01 
    simulationTime = 25
    simulationStep = int(simulationTime / simulationDt) 
    controllerSpeedNR = 35
    lookaheadTime = 0.20
    

    invertedPendulum = InvertedPendulum(initialX1, initialX2, lookaheadTime = lookaheadTime, odeStepSize= 0.01*lookaheadTime)

    inputList = [0]
    referenceSignal = [inputSignal(0)]
    for curStep in range(1, simulationStep):
        predictionX1, predictionX2 = invertedPendulum.predictSystem(inputList[-1])
        derivative = invertedPendulum.getPredictionDerivative(inputList[-1])
        derivative = invertedPendulum.getPredictionDerivativeFiniteDifference(inputList[-1])
        
        predictionSignal = inputSignal(curStep * simulationDt + lookaheadTime)
        invertedPendulum.runSystem([inputList[-1]])
       
        du = controllerSpeedNR * (predictionSignal - predictionX1) / derivative 
        
        inputList.append(inputList[-1] + du * simulationDt)
        referenceSignal.append(inputSignal(curStep * simulationDt))

    plt.plot(invertedPendulum.stateX1History,  label = 'System output')
    plt.plot(referenceSignal, '--', label = 'Reference signal')
    absError = np.abs(np.array(invertedPendulum.stateX1History) - np.array(referenceSignal))
    
    plt.grid()
    mseError = np.sum(absError ** 2)/absError.shape[0]
    plt.legend()
    plt.xlabel("Sample index (0.01s)")
    plt.ylabel("Outputs (rad)")
    plt.title("Tracking result using model-based method, inverted pendulum\n Lookahead time = %.2fs, MSE = %.4f"%(lookaheadTime, mseError))
    plt.figure()
 
    plt.grid()
    plt.title("u(t) using model-based method, inverted pendulum\n Lookahead time = %.2fs"%(lookaheadTime))
    plt.plot(inputList, label = "System input u(t)")
    plt.xlabel("Sample index (0.01s)")
    plt.ylabel("u(t)")
    plt.legend()
    plt.show()

        


