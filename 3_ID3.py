'''
Write a program to demonstrate the working of the decision tree based ID3
algorithm. Use an appropriate data set for building the decision tree and apply this
knowledge to classify a new sample.
'''

import math
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, data, attribute=None):
        self.child = {}
        self.data = data
        self.decision = None
        self.decision_attribute = attribute
        
def calculate_entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])

def split_data(l, attribute, Class):
    return l[l[attribute] == Class]

def entropy(l, attribute = 'PlayTennis', Gain = False):
    counter = Counter(l[attribute])
    n = len(l[attribute])
    probs = [x / n for x in counter.values()]
    if not Gain:
        return calculate_entropy(probs)
    gain = 0
    for Class, prob in zip(counter.keys(), probs):
        gain += -prob * entropy(split_data(l, attribute, Class))
    return gain

def information_gain(data):
    max_gain = -1
    max_gain_attribute = None
    for attr in data.keys():
        if attr == 'PlayTennis':
            continue
        gain = entropy(data) + entropy(data, attr, Gain=True)
        if gain > max_gain:
            max_gain = gain
            max_gain_attribute = attr
    return max_gain_attribute

def id3(root):
    global nodes
    if len(root.data.keys()) == 1 or len(root.data) == 1:
        counter = Counter(root.data['PlayTennis'])
        root.decision = counter.most_common(1)[0][0]
        return
    max_gain_attribute = information_gain(root.data)
    root.decision_attribute = max_gain_attribute
    for attr in set(root.data[max_gain_attribute]):
        child = split_data(root.data, max_gain_attribute, attr)
        root.child[attr] = Node(child.drop([max_gain_attribute], axis=1))
        id3(root.child[attr])

def predict(example, root):
    if root.decision != None:
        return root.decision
    try:
        pred = predict(example, root.child[example[root.decision_attribute]])
        return pred
    except:
        return "No"
        
df = pd.DataFrame.from_csv('3_tennis.csv')
df = df.sample(frac=1).reset_index(drop=True)
train = df.iloc[:-4]
test = df.iloc[-4:]
root = Node(data = train)
id3(root)
print(test)
pred = [predict(t, root) for _,t in test.iterrows()]
correct = test['PlayTennis']
print("Predictions: {}".format(pred))
print("Actual: \n{}".format(correct))
print("Accuracy: {}".format(sum([1 for x,y in zip(pred, correct) if x==y]) / (len(pred))))