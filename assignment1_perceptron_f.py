# Written by Austen Hsiao for Assignment 1, cs545 (MachineLearning)
# PSU ID: 985647212
# This program has no error checking and I did not write unit tests.
# It's hardcoded for the files for assignment 1

import numpy as np
import pandas as pd
import os 
import random
import time

class Network:
	# Upon object instantiation, the weights matrix is created. 
	# Each individual array in the matrix are perceptron weights
	weights = np.random.uniform(low=-0.05, high=0.05, size=(10,785))
	learningRate = 0
	
	# Constructor allows us to set the learning rate
	def __init__(self, rate):
		self.learningRate = rate

	# opens "mnist_train.csv". Adds a column of 1s at the end (this will be the bias input.
	# Then divides columns 1:784 by 255, then writes these values to "scaledTS.csv, so we don't alter the original training set data
	def setUpTrainingSet(self):
		trainingSet = pd.read_csv("mnist_train.csv", header=None)
		if os.path.isfile("scaledTS.csv"):
			os.remove("scaledTS.csv")
		print("Creating scaledTS.csv. May take about a minute")
		trainingSet.insert(785,785,1) # insert a column of 1s at column 785 for all rows
		first = trainingSet.loc[0:60000, 0:0] # first part is column 0 of all rows
		middle = trainingSet.loc[0:60000, 1:784].apply(lambda x: x/255) # middle part is columns 1:784 for all columns
		end = trainingSet.loc[0:60000, 785:785] # end part is the column of ones
		# I split up the document like this so I can apply /255 to the center columns
		(first.join(middle)).join(end).to_csv("scaledTS.csv", mode="a", header=False, index=False) # Here is where I join all the pieces together and write it to scaledTS.csv
		print("Done")
		return	

	# generates a new csv (accuracy.csv) with the accuracies for the given epoch-- for both trainingSet and validationSet
	# This uses the current weights, so we better specify the correct epochNumber!
	def reportAccuracy(self, currentEpoch, trainingSet, validationSet):
		trainingAcc = [0]*10
		validationAcc = [0]*10	
	
		# creates an array of outputs for each perceptron, then reports the index of the highest value
		# if this value is the same as the expected perceptron, increase acc
		acc = 0
		for line_of_data in trainingSet:
			for perceptron in range(10):
				trainingAcc[perceptron] = np.dot(line_of_data[1:], self.weights[perceptron])
			calculatedPerceptron = trainingAcc.index(max(trainingAcc))
			if calculatedPerceptron == line_of_data[0]:
				acc += 1
		trainingAccuracy = acc/len(trainingSet)

		acc = 0
		for line_of_data in validationSet:
			for perceptron in range(10):
				validationAcc[perceptron] = np.dot(line_of_data[1:], self.weights[perceptron])
			calculatedPerceptron = validationAcc.index(max(validationAcc))
			if calculatedPerceptron == line_of_data[0]:
				acc += 1
		validationAccuracy = acc/len(validationSet)

		print("Training accuracy for epoch", currentEpoch, ":", trainingAccuracy)
		print("Validation accuracy for epoch", currentEpoch, ":", validationAccuracy)
		pd.DataFrame({'epoch':[currentEpoch],'training':[trainingAccuracy],'validation':[validationAccuracy]}, columns=['epoch','training','validation']).to_csv("accuracy.csv", mode='a', header=False, index=False)
		return
	
	# calculatedOutput returns the results for given perceptron. These are not using the Sigmoid function, so the result is either 0 or 1.
	def calculatedOutput(self, data, perceptronNumber):
		if np.dot(data[1:], self.weights[perceptronNumber]) > 0:
			return 1
		return 0
	
	# expectedOutput returns the expected result for a given perceptron, used during training.
	def expectedOutput(self, data, perceptronNumber):
		if data[0] == perceptronNumber:
			return 1
		return 0
	
	# Trains specified perceptron with the given data. Updates all weights
	def trainPerceptron(self, data, perceptronNumber, expected, initialCalculated):
		y = initialCalculated
		
		# 'weight' is a bit misleading, it's just the numbers [0,784].
		# Checking the data starts at weight+1 because the 0th column is the class identifier
		for weight in range(785):
			if data[weight+1] == 0:
				continue
			self.weights[perceptronNumber][weight] += self.learningRate*(expected - y)*data[weight+1]
			# after weight is changed, check output difference. We stop if it matches expected
			y = self.calculatedOutput(data, perceptronNumber)
			if y == expected:
				return

	# run numEpoch number of epochs
	def runEpoch(self, numEpoch):
		if os.path.isfile("accuracy.csv"):
			os.remove("accuracy.csv")
		trainingSet = pd.read_csv("scaledTS.csv", header=None).to_numpy()
		validationSet = pd.read_csv("mnist_validation.csv", header=None)
		# need to append the column of 1s to the validation set
		validationSet.insert(785,785,1)
		validationSet = validationSet.to_numpy()

		start = time.time()
		# report the accuracy for the 0th epoch (calling this function writes to a csv)
		self.reportAccuracy(0, trainingSet, validationSet)	
		print("Time:", time.time()-start)

		# for all the epochs specified, we shuffle the trainingSet
		# Then we iterate through each training set example, applying it to each of the perceptrons.
		# Once all examples are run, we report the accuracy, writing it to a file.
		for epoch in range(numEpoch):
			np.random.shuffle(trainingSet)
			start = time.time()
			for data in trainingSet:
				for perceptron in range(10):
					t = self.expectedOutput(data, perceptron)
					y = self.calculatedOutput(data, perceptron)
					if t == y:
						continue
					self.trainPerceptron(data, perceptron, t, y)
			self.reportAccuracy(epoch+1, trainingSet, validationSet)
			print("Time:", time.time()-start)

		return

	# returns the class when run through the network		
	def returnCalculatedClassForTrial(self, single_trial):
		output = [0]*10
		for perceptron in range(10):
			output[perceptron] = np.dot(single_trial[1:], self.weights[perceptron])
		
		return output.index(max(output))
		
	# creates a confusion matrix (rows: actual, columns: predicted)
	def confusionMatrix(self):
		validationSet = pd.read_csv("mnist_validation.csv", header=None)

		validationSet.insert(785,785,1)
		validationSet = validationSet.to_numpy()

		cMatrix = np.zeros((10,10))

		if os.path.isfile("confusion_matrix.csv"):
			os.remove("confusion_matrix.csv")
		for trial in validationSet:
			cMatrix[self.returnCalculatedClassForTrial(trial)][trial[0]] += 1

		pd.DataFrame(data=cMatrix).to_csv("confusion_matrix.csv", mode='a', header=False, index=False)
		return

if __name__== '__main__':
	test = Network(0.00001)
	test.setUpTrainingSet()
	test.runEpoch(50)
	test.confusionMatrix()
