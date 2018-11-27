'''
Implement and demonstratethe FIND-S algorithm for finding the most specific
hypothesis based on a given set of training data samples. Read the training data from a
.CSV file.

'''
import csv

a = []
num_attributes = 6

print("\n The Given Training Data Set \n")
with open('1_training-examples.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)
print()

# Comparing with First Training Example 
for j in range(0,num_attributes):
        hypothesis[j] = a[0][j];

# Comparing with Remaining Training Examples of Given Data Set
for i in range(0,len(a)):
    if a[i][num_attributes]=='Yes':
            for j in range(0,num_attributes):
                if a[i][j]!=hypothesis[j]:
                    hypothesis[j]='?'
                else :
                    hypothesis[j]= a[i][j] 
    print(hypothesis)
                
print("\n The Maximally Specific Hypothesis for a given Training Examples :")
print(hypothesis)