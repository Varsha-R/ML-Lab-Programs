'''
Write a program to construct aBayesian network considering medical data. Use this
model to demonstrate the diagnosis of heart patients using standard Heart Disease
Data Set.

'''
import numpy as np
import pandas as pd
from urllib.request import urlopen

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']

data = pd.read_csv(urlopen(url), names=names)
del data['ca']
del data['slope']
del data['thal']
del data['oldpeak']
data = data.replace('?', np.nan)

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), 
                       ('exang', 'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),
                      ('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])

# Learing CPDs using Maximum Likelihood Estimators 
model.fit(data, estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
print(model.get_cpds('sex'))
print(model.get_cpds('chol'))
model.get_independencies()

# Doing exact inference using variable elimination
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# Computing probability of bronc given smoke
q = infer.query(variables=['heartdisease'], evidence={'age':28})
print(q['heartdisease'])
q = infer.query(variables=['heartdisease'], evidence={'chol':100})
print(q['heartdisease'])