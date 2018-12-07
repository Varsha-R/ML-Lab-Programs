'''
Write a program to implement the naïve Bayesian classifier for a sample training
data set stored as a .CSV file. Compute the accuracy of the classifier, considering few
test data sets.
'''

import csv
import math
import random

def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    data = list(lines)
    for i in range(len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def split_dataset(data, splitRatio):
    size = int(len(data) * splitRatio)
    training = []
    testing = list(data)
    while len(training) < size:
        index = random.randrange(len(testing))
        training.append(testing.pop(index))
    return [training, testing]

def separate_by_class(data):
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(data):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
    del summaries[-1]
    return summaries

def summarize_by_class(data):
    separated = separate_by_class(data)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculate_class_probabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def get_predictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def get_accuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


filename = '5_pima-indians-diabetes.csv'
splitRatio = 0.67
data = load_csv(filename)
    
train, test = split_dataset(data, splitRatio)
summaries = summarize_by_class(train)
print("Model summaries: ", summaries)
predictions = get_predictions(summaries, test)
print("Predictions: ", predictions)
accuracy = get_accuracy(test, predictions)
print("Accuracy: ", accuracy)