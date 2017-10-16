# -*- coding: utf-8 -*-
'''
Created on 10.10.2017

@author: Leonard
'''
from numpy import *
import os
import pickle
import logging



sigmoid = lambda x : 1/(1+exp(-x))
sigmoid_deriv = lambda x : exp(-x)/((1+exp(-x))**2)

ReLU = lambda x : maximum(0,x)
ReLU_deriv = lambda x : (sign(x)+1)/2.0

linear = lambda x : x
linear_deriv = lambda x : 1

softsign = lambda x : x/(1.0+abs(x))
softsign_deriv = lambda x : 1/(1.0+abs(x))**2

activationFunction = sigmoid
activationFunction_deriv = sigmoid_deriv

start_weight_range = 0.5

class NeuralNetwork(object):
    '''
    Neural Network
    '''


    def __init__(self, layerDimensions, dirName):
        
        self.directory = dirName
        
        if not os.path.exists(dirName):
            os.mkdir(dirName)

        self.layerDimensions = layerDimensions
        self.layers = len(layerDimensions)
        self.weightMatrices = []
        self.biases = []
       # self.activationFunction = sigmoid
        
        for i in range(len(layerDimensions)-1):
            #weightMatrix = matrix(array(zeros((layerDimensions[i+1],layerDimensions[i]))))
            #weightMatrix.fill(start_weight)
            weightMatrix = matrix(random.rand(layerDimensions[i+1],layerDimensions[i]))
            weightMatrix = (weightMatrix-0.5)*2*start_weight_range
            self.weightMatrices.append(weightMatrix)
        for layerDim in layerDimensions:
            self.biases.append(zeros(layerDim))
            
        self._neuronInputs = [] # netzeingabe der neuronen
        self._neuronValues = [] # ausgabe eines neurons: aktivierungsfunction(_neuronInputs)
        self._targetOutputNeurons = None
        self._errorSignals = None
        
        self.learnRate = 0.1
        self.bias_learnRate = 0
        
        self.Save()
        
    def setWeight(self, weightLayer, in_index, out_index, value):
        self.weightMatrices[weightLayer][out_index,in_index] = value
    def getWeight(self, weightLayer, in_index, out_index):
        return self.weightMatrices[weightLayer][out_index,in_index]
    
    def Train(self, inputLayer, correctOutputLayer):
        outputLayer = self.feedForward(inputLayer)
        self._targetOutputNeurons = correctOutputLayer
        self._errorSignals = []
        for arr in self._neuronInputs:
            k = array(zeros(len(arr)), dtype = object)
            k.fill(None)
            self._errorSignals.append(k)
            
            
        # update weights
        changed_weights = 0
        
        new_weights = [None]*len(self.weightMatrices)
        for weight_matrix_index in range(len(self.weightMatrices)-1,-1,-1):
            weight_matrix = self.weightMatrices[weight_matrix_index]
            shape = weight_matrix.shape
            deltaMatrix = matrix(array(zeros(shape)))
            for inputNeuronIndex in range(shape[1]):
                for outputNeuronIndex in range(shape[0]):
                    e = self.getErrorSignal(weight_matrix_index+1, outputNeuronIndex)
                    
                    d = -self.learnRate*e*self._neuronValues[weight_matrix_index][inputNeuronIndex]

                 #   logging.debug("Neuron value at {0},{1}: ".format(weight_matrix_index,inputNeuronIndex)+str(self._neuronValues[weight_matrix_index][inputNeuronIndex]))
                  #  logging.debug("Error signal at {0},{1}: ".format(weight_matrix_index+1,outputNeuronIndex)+str(e))
                    deltaMatrix[outputNeuronIndex,inputNeuronIndex] = d
                    if (d!=0):
                        changed_weights += 1
            newMatrix = weight_matrix+deltaMatrix
            new_weights[weight_matrix_index] = newMatrix
            
        self.weightMatrices = new_weights
        
        # update biases
        changed_biases = 0
        for layer in range(1,self.layers):
            for j in range(self.layerDimensions[layer]):
                d = -self.bias_learnRate*self._errorSignals[layer][j]
                if d!=0:
                    changed_biases += 1
                    self.biases[layer][j] += d
        
        #print("Finished training: {0} weights and {1} biases were changed".format(changed_weights, changed_biases))
        
    def getErrorSignal(self, neuronLayer, neuronIndex): # ∂E/∂o_j * ∂o_j/∂net_j
        if self._errorSignals[neuronLayer][neuronIndex] != None:
            return self._errorSignals[neuronLayer][neuronIndex]
        o_j = self._neuronValues[neuronLayer][neuronIndex]
        if neuronLayer == self.layers-1: # outputneuron
            #(o_j-t_j)*o_j*(1-o_j)
            t_j = self._targetOutputNeurons[neuronIndex]
            dE_doj = o_j-t_j
           # dE_doj = (o_j-t_j)/(o_j*(1-o_j))
            val = dE_doj*o_j*(1-o_j)
            #print("Error signal in output-layer at {0}: {1}, Neuron value: {2},Target Neuron Value: {3}".format(neuronIndex,val,o_j, t_j))
            self._errorSignals[neuronLayer][neuronIndex] = val
            return val
        else:
            s = 0
            L = range(self.layerDimensions[neuronLayer+1])
            for l in L:
                err_sig = self.getErrorSignal(neuronLayer+1, l)
                w_jl = self.getWeight(neuronLayer, neuronIndex, l)
                s += err_sig*w_jl
            val = s*o_j*(1-o_j)
            self._errorSignals[neuronLayer][neuronIndex] = val
            return val
        """netInput = self._neuronInputs[neuronLayer][neuronIndex]
        if neuronLayer == self.layers-1: # neuron ist in der ausgabeschicht
            val = activationFunction_deriv(netInput)*(self._neuronValues[neuronLayer][neuronIndex]-self._targetOutputNeurons[neuronIndex])
            print("Error signal in output-layer at {0}: {1}, Neuron value: {2},Target Neuron Value: {3},NetInput: {4}".format(neuronIndex,val,self._neuronValues[neuronLayer][neuronIndex],self._targetOutputNeurons[neuronIndex],netInput))
            self._errorSignals[neuronLayer][neuronIndex] = val
            return val
        else:
            s = 0
            for nextNeuronIndex in range(self.layerDimensions[neuronLayer+1]):
                s += self.getErrorSignal(neuronLayer+1, nextNeuronIndex)*self.getWeight(neuronLayer, neuronIndex, nextNeuronIndex)
            val = activationFunction_deriv(netInput)*s
            self._errorSignals[neuronLayer][neuronIndex] = val
            return val"""
                
        
    def feedForward(self, inputLayer):
        self._neuronValues = []
        self._neuronInputs = []
        if len(inputLayer) != self.layerDimensions[0]:
            raise Exception("Inputlayer length != first layer length in data")
        
        currentLayer = inputLayer
        
        self._neuronInputs.append(currentLayer)
        
        for layer in range(1,len(self.layerDimensions)):
            self._neuronValues.append(array(currentLayer))
            
            weights = self.weightMatrices[layer-1]
            biases = self.biases[layer]
            
            neuronInputs = dot(weights,currentLayer)-biases
            #print(neuronInputs)

            #self._neuronInputs.append(array(array(neuronInputs)[0]))
            self._neuronInputs.append(neuronInputs.tolist()[0])
            
            nextLayer = activationFunction(neuronInputs)
            currentLayer = nextLayer.tolist()[0]
            
        self._neuronValues.append(array(currentLayer))
        return currentLayer

    def Save(self):
        f = open(self.directory+"/network.net",'w')
        pickle.dump(self,f)
        f.close()
        
        
        
def loadNetwork(directory):
    f = open(directory+"/network.net",'r')
    net = pickle.load(f)
    f.close()
    return net

def testNeuralNetwork(): # XOR
    network = NeuralNetwork([2,4,2],"TestNetwork2")
    for i in range(1):
        network.Train(array([0,0]), array([0,1]))
        network.Train(array([1,0]), array([1,0]))
        network.Train(array([0,1]), array([1,0]))
        network.Train(array([1,1]), array([0,1]))

    network.Save()
    
    inputLayer = array([1,1])
    print(network.feedForward(inputLayer))
    #print(network.weightMatrices[0])
if __name__ == "__main__":
    from NumberRecognition import main
    main()
    #testNeuralNetwork()
   # inputLayer = array([2,1,.1])
   # print(loadNetwork("TestNetwork").feedForward(inputLayer))