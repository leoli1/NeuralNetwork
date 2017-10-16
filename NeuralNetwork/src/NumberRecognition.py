'''
Created on 11.10.2017

@author: Leonard
'''
from PIL import Image
from NeuralNetwork import *
from numpy import *
import matplotlib.pyplot as plt
import logging
import atexit

class TrainingPattern:
    def __init__(self, pattern, number):
        self.inputPattern = pattern
        self.number = number
network = None
trainingPatterns = []
testPatterns = []

trainingData = "NumberRecognition/TrainingData/"
testData = "NumberRecognition/TestData/"

#networkName = "NumberRecognition5Layer"
networkName = "NumberRecognition8Layer2"
#networkName = "NumberRecognition6Layer"

def CreateNetwork():
    global network
    
    #network = NeuralNetwork([1024,32,16,16,10],networkName)
    network = NeuralNetwork([1024,512,256,128,64,32,16,10],networkName)
    #network = NeuralNetwork([1024,64,32,16,16,10],networkName)
    #network = NeuralNetwork([1024,10],"NumberRecognition")
    
    
def LoadTrainingData():
    for i in range(10):
        for f in os.listdir(trainingData+str(i)):
            if f.endswith(".png"):
                trainingPatterns.append(TrainingPattern(getImgArray(trainingData+str(i)+"/"+f), i))#getImgArray(trainingData+str(i)+"/"+f))
def LoadTestData():
    for i in range(10):
        for f in os.listdir(testData+str(i)):
            if f.endswith(".png"):
                testPatterns.append(TrainingPattern(getImgArray(testData+str(i)+"/"+f), i))#getImgArray(trainingData+str(i)+"/"+f))
      

def TrainNetwork(numbers=range(10),iters=1):
    print("----START TRAINING----")
    for k in range(iters):
        print("Train cycle: {0}".format(k))
        for pat in trainingPatterns:
            TrainImage(pat)

            
    network.Save()
    
def TestNetwork(numbers=range(10)):
    print("----START TESTING----")
    r = 0
    for pat in testPatterns:
        out = network.feedForward(pat.inputPattern)
        o = intNetworkOutput(out)
        print(("Network output is: {0}, expected output is: {1}".format(o,pat.number)))
        if o == pat.number:
            r += 1
        #print(array(network._neuronInputs))
        
    print("{0} von {1} richtig erkannt".format(r,len(testPatterns)))
    network.Save()
    
def intNetworkOutput(arr):
    return array(arr).argmax(0)
    
def getImgArray(fileName):
    img = asarray(Image.open(fileName))

    conv_img = ((255-img)/255.0)[:,:,:1]
    conv_img[conv_img>0] = 0.99
    conv_img[conv_img==0] = 0.01
    return array(conv_img).reshape(-1,)

def TrainImage(pattern):
    conv_img = pattern.inputPattern
    output = [0.01]*10
    output[pattern.number] = 0.99
    network.Train(conv_img, output)
    
    
createNew = False
trainNetwork = False
trainIterations = 750
def main():
    global network
    #logging.basicConfig(filename='trainlog2.log',level = logging.DEBUG)
    LoadTrainingData()
    LoadTestData()
    if createNew:
        CreateNetwork()
    else:
        network = loadNetwork(networkName)
    if trainNetwork:
        TrainNetwork(iters=trainIterations)#numbers=[4,5,6])
        
    TestNetwork()#numbers=[4,5,6])
    
    #TrainImage("black.png", 1)
    
    for i in range(network.layers-1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im  = ax.matshow(network.weightMatrices[i], aspect=network.layerDimensions[i]/network.layerDimensions[i+1])
        ax.set_title("weight matrix: "+str(i))
        fig.colorbar(im)
        
    plt.show()
    
if __name__ == '__main__':
    main()