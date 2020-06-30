# Perceptron rule- Assignment1, written by Austen Hsiao for CS545
import numpy as np
import pandas as pd
import os
import random
import time

### Macros ###
ROW = 10
COLUMN = 785
FILE_ROW = 60000
FILE_COLUMN = 785

class Network:
    weights = np.zeros((ROW, COLUMN))
    learningRate = 0

    # Constructor, takes in an int for the learningRate. 
    # Creates a ROWxCOLUMN matrix populated with random weights in the interval [-0.05 and 0.05)
    def __init__(self, lrate):
        self.learningRate = lrate
        for i in range(ROW):
            for j in range(COLUMN):
                self.weights[i][j] = random.uniform(-0.05, 0.05)

    # Prints out the weight matrix-- mainly used to make sure the constructor set things up right
    def printWeights(self):
        for i in self.weights:
            print(i)
        return
    
    # Should be called without arguments, since it is supplied by default.
    # Opens "mnist_train.csv" and normalizes all values except for the expected results.
    # Also adds another column for the bias input. 
    # These rows are then written to "scaledTS.csv". 
    # I decided to write these values to a new csv so as to not destroy/alter the original training set file.
    # This method takes a while to run, but all subsequent runs won't be as slow.
    # Hardcoded for 60000x785 csv
    def open_mnist_train_and_initialize_trainingSet(self, name="mnist_train.csv"):
        try: 
            trainingSet = pd.read_csv(name, header=None)
            if os.path.isfile("scaledTS.csv"):
                os.remove("scaledTS.csv")
        except:
            print("filename", name, "not found")
            return -1
        # This 1, inserted at the end of the pre-existing columns is for the bias
        trainingSet.insert(785, 785, 1)
        print("Setting up scaled and randomly ordered training set. This involves converting Ints to Floats and may take a while. Please wait...")
        ((trainingSet.loc[0:20000, 0:0].join(trainingSet.loc[0:20000, 1:784].apply(lambda x: x/255))).join(trainingSet.loc[0:20000, 785:785])).to_csv("scaledTS.csv", mode="a", header=False, index=False)
        ((trainingSet.loc[20001:40000, 0:0].join(trainingSet.loc[20001:40000, 1:784].apply(lambda x: x/255))).join(trainingSet.loc[20001:40000, 785:785])).to_csv("scaledTS.csv", mode="a", header=False, index=False)
        ((trainingSet.loc[40001:59999, 0:0].join(trainingSet.loc[40001:59999, 1:784].apply(lambda x: x/255))).join(trainingSet.loc[40001:59999, 785:785])).to_csv("scaledTS.csv", mode="a", header=False, index=False)    
        print("Done.")
        return 1

    # Takes one int as input-- the number of epochs to run. This function will open "scaledTS.csv" and run
    # epocs, training the perceptrons along the way. Data for accuracy is recorded and appended to a file, "accuracy.csv".
    # Returns 1 if it runs to completion. It also prints out run times in seconds for each epoch. 
    def run_epoch(self, numberofEpochs):
        if not os.path.isfile("scaledTS.csv"):
                print("Run open_mnist_train_and_initialize_trainingSet() before running this function! scaledTS.csv not found")
                return -1
        if os.path.isfile("accuracy.csv"):
            os.remove("accuracy.csv")

        trainingSet = pd.read_csv("scaledTS.csv", header=None)
        trainingSet = trainingSet.to_numpy()
        np.random.shuffle(trainingSet)

        start = time.time()
        accuracy = [0, self.compute_accuracy(trainingSet)]
        pd.DataFrame(accuracy).to_csv("accuracy.csv", mode='a', header=False, index=False)
        print("Initial accuracy:", accuracy[1])
        print("Time:", time.time() - start)
        
        for i in range(numberofEpochs):
            np.random.shuffle(trainingSet) 
            start = time.time()
            for single_trial in trainingSet:
                for perceptronNumber in range(10):
                    calculatedOutput = self.calculated_output(single_trial, perceptronNumber)
                    expectedOutput = self.expected_output(single_trial, perceptronNumber)
                    self.run_one_trial(single_trial, perceptronNumber, calculatedOutput, expectedOutput)
            
            accuracy = [i+1, self.compute_accuracy(trainingSet)]
            pd.DataFrame(accuracy).to_csv("accuracy.csv", mode='a', header=False, index=False)
            print("Accuracy for epoch", i+1, ": ", accuracy[1])
            print("Time:", time.time() - start)
        return 1
        
    # Computes the accuracy for a given set of inputs formatted as [A, x, x, x, x..., 1] where 
    # A = expected output class, x's are inputs, and '1' is the bias input.
    # Returns float
    def compute_accuracy(self, inputs):         
        acc = 0
        total = len(inputs)
        for trial in inputs:
            classifier, high = -1, -9999999
            for perceptron in range(10):
                output = np.dot(trial[1:], self.weights[perceptron])
                if output > high:
                    high = output
                    classifier = perceptron
            if classifier == int(trial[0]):
                acc += 1
        return acc/float(total)

    # returns 0 or 1 for a single trial (training data) and perceptron number,
    # will return 1 if the given perceptron will fire based on the data.
    # 'inputs' is formated consistently with the rest of this script, that is:
    # [A, x, x, x, x..., 1] where A = expected output class, x's are inputs, and '1'
    # is the bias input. 
    def calculated_output(self, single_trial, perceptronNumber):
        if np.dot(single_trial[1:], self.weights[perceptronNumber]) > 0:
            return 1
        return 0

    # returns 0 or 1 for a single trial (training data) and perceptron number,
    # will return 1 if the given perceptron is supposed to fire.
    # 'inputs' is formated consistently with the rest of this script, that is:
    # [A, x, x, x, x..., 1] where A = expected output class, x's are inputs, and '1'
    # is the bias input.
    def expected_output(self, single_trial, perceptronNumber):
        if single_trial[0] == perceptronNumber:
            return 1
        return 0

    # Inputs: one line of data ([A, x, x, x, x..., 1]), perceptron number, the calculated output for the data, and the 
    # expected output for the data. This function will update the weight matrix by calculating deltaW, as explained by the 
    # perceptron training algorithm. return type void
    def run_one_trial(self, trainingData, perceptronNumber, calculatedOutput, expectedOutput):
        if calculatedOutput == expectedOutput:
            return
        for training_index in range(len(trainingData[1:])):
            if trainingData[training_index] == 0:
                continue
            delW = self.learningRate * (expectedOutput - calculatedOutput) * trainingData[training_index]
            self.weights[perceptronNumber][training_index] += delW
        return 
    
    # Takes 2 arguments, but should really only be run with 1 argument. Specify 1 if the perceptrons have already been trained,
    # otherwise, any other number works. This function opens up "mnist_validation.csv" and runs the accuracy function upon it.
    # Again, all of the data in the validation csv are formatted identically to the training set csv. Returns 1 if ran to completion 
    def check_validation_set(self, after=1, name="mnist_validation.csv"):
        try: 
            validationSet = pd.read_csv(name, header=None)
        except:
            print("mnist_validation.csv not found")
            return -1
        validationSet.insert(785, 785, 1)
        validationSet = validationSet.to_numpy()
        accuracy = self.compute_accuracy(validationSet)
        if after == 1:
            print("Validation set accuracy after training:", accuracy)
            return 1
        print("Validation set accuracy before training:", accuracy)
        return 1
        
if __name__ == '__main__':
    assignment1 = Network(0.1)
    assignment1.check_validation_set(0)
    assignment1.run_epoch(3)
    assignment1.check_validation_set(1)