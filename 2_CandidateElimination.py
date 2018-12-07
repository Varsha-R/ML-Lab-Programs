'''
For a given set of training data examples stored in a .CSV file, implement and
demonstrate the Candidate-Elimination algorithm to output a description of the set
of all hypotheses consistent with the training examples.
'''

import csv

def domain():
    D = []
    for i in range(len(data[0])):
        D.append(list(set([ele[i] for ele in data])))
    return D

def consistent(h1, h2):
    for x, y in zip(h1, h2):
        if not (x == '?' or (x!='&' and (x==y or y=='&'))):
            return False
    return True

def candidate_elimination():
    G = {('?',)*(len(data[0]) - 1),}
    S = ['&']*(len(data[0]) - 1)
    n = 0
    print("\n G[{0}]:".format(n), G)
    print("\n S[{0}]:".format(n), S)
    
    for item in data:
        n+=1
        inp, res = item[:-1], item[-1]
        if res in "Yy":
            i = 0
            G = {g for g in G if consistent(g, inp)}
            for s,x in zip(S, inp):
                if not s==x:
                    S[i] = '?' if s!='&' else x
                i+=1
        else:
            S = S
            Gprev = G.copy()
            for g in Gprev:
                if g not in G:
                    continue
                for i in range(len(g)):
                    if g[i] == '?':
                        for val in D[i]:
                            if inp[i]!=val and val==S[i]:
                                g_new = g[:i] + (val,) + g[i+1:]
                                G.add(g_new)
                    else:
                        G.add(g)
                G.difference_update([h for h in G 
                                     if any([consistent(h, g1) for g1 in G if h!=g1])])
        
        print("\n G[{0}]:".format(n), G)
        print("\n S[{0}]:".format(n), S)
        

with open('2_trainingexamples.csv') as csvFile:
    data = [tuple(line) for line in csv.reader(csvFile)]
    
D = domain()
candidate_elimination()
        
