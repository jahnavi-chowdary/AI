import random
import argparse, os, sys
import pandas as pd
import numpy as np
import copy
import time
import csv
import pickle as pkl
import scipy.stats as stats
from scipy.stats import chisquare
import math

sys.setrecursionlimit(100000)

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''
# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# sys.setrecursionlimit(100000)

# No.of Internal Nodes
internal_nodes = 0

# No.of Leaf Nodes
leaf_nodes = 0

# Load the Train and Test Data from the files specified by the User
def load_data(ftrain, ftest):
    Xtrain = pd.read_csv(ftrain, header=None, delim_whitespace=True)
    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    Ytrain = pd.read_csv(ftrain_label, header=None, delim_whitespace=True)

    Xtest = pd.read_csv(ftest, header=None, sep=" ")

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest

# Function to calculate the p-value
def calculate_chi():
    obs = []
    exp = []
    c, p = chisquare(obs, exp)
    return p

# Evaluate and return the chi square p value
def chi2_func(Xtrain, best_attribute):    
    obs = []
    exp = []
    
    # Number of -ve samples    
    n = (Xtrain['target'] == 0).sum() 
    # number of +ve samples
    p = (Xtrain['target'] == 1).sum()
    
    N = n + p    
    r1 = float(p) / N
    r2 = float(n) / N    
    
    # For each value calculate the expected and observed number of +ve and -ve
    for value in Xtrain[best_attribute].unique():                
        per_attr = Xtrain.filter([best_attribute,'target'],axis=1)
        per_attr = per_attr.loc[(per_attr[best_attribute]==value)]
        
        Ti = per_attr['target'].count() 
        pi = float((per_attr['target'] == 1).sum())
        ni = float((per_attr['target'] == 0).sum())
        
        k = pi * ni                                       
        pi1 = float(r1) * Ti
        ni1 = float(r2) * Ti

        if ni1 != 0:
            exp.append(ni1)        
            obs.append(ni) 
        if pi1 != 0:
            exp.append(pi1)
            obs.append(pi)

    # Calculate the chi-square value
    c, p = chisquare(obs, exp)

    # Return the p value
    return p

# Decide the best attribute to split on based on the Maximum Entropy Gain

def getValues(self, Xtrain, i):
        array = []
        for sample in Xtrain:
            val = sample[i]
        return val

def OutputEntropy(self, Datalabel):
        pos = 0.0
        neg = 0.0
        total = len(Datalabel)
        for i in Datalabel:
            if i == 0:
                neg += 1
            elif i == 1:
                pos += 1
        ans_entropy = 0
        if pos != 0:
            ans_entropy -= (pos/total) * log(pos/total, 2)
        if neg != 0:
            ans_entropy -= (neg/total) * log(neg/total, 2)
        return ans_entropy

def calculate_split(node, attributes):
    if node.data == 'T':
        k = 1
    elif node.data == 'F':
        k = 0
    return k

def find_attribute_to_split_on(Xtrain, attributes):    
    entropy_min = None
    best_attribute = None    

    # For each attribute, calculate the gain and the attribute with maximum gain
    for attribute in attributes:                 
        rows_count = Xtrain[attribute].count()
        entropy = 0   

        for value in Xtrain[attribute].unique():
            count = (Xtrain[attribute] == value).sum()
            p = float(count) / rows_count            
            
            per_attr = Xtrain.filter([attribute, 'target'], axis=1)            
            per_attr = per_attr.loc[(per_attr[attribute] == value)]
            per_attr_rows_count = per_attr['target'].count()            
            pos = (per_attr['target'] == 1).sum()
            prob_pos = float(pos) / per_attr_rows_count
            neg = (per_attr['target'] == 0).sum()
            prob_neg = float(neg) / per_attr_rows_count
            if prob_pos == 0:
                entropy_pos = 0
            else:
                entropy_pos = prob_pos * (np.log2(prob_pos))
            if prob_neg == 0:
                entropy_neg = 0
            else:
                entropy_neg = prob_neg * (np.log2(prob_neg))                            
            entropy_total = -(entropy_neg + entropy_pos)        
            entropy += p * entropy_total
        if entropy_min == None or entropy < entropy_min:
            best_attribute = attribute
            entropy_min = entropy 
    # return the best attribute   
    return best_attribute

def sample_space_func(Xtrain, best_attribute):
    sample_space_subset = Xtrain.loc[Xtrain[best_attribute] == i]
    l = 0
    if sample_space_subset.empty:
        pos = (Xtrain['target'] == 1).sum()
        neg = (Xtrain['target'] == 0).sum()
        if pos >= neg:
            l += 1
        else:
            l -= 1

def create_tree(Xtrain, attributes, pval):
    global internal_nodes, leaf_nodes
    
    # Base Case : If the Train sample has only +ve values, return the node with 'True'   
    if (Xtrain['target'] == 1).sum() == Xtrain['target'].count():
        leaf_nodes += 1 # Increment no.of leaf_nodes
        return TreeNode()        
    
    # Base Case : If the Train sample has only -ve values, return node with 'False'
    if (Xtrain['target'] == 0).sum() == Xtrain['target'].count():
        leaf_nodes += 1 # Increment no.of leaf_nodes
        return TreeNode('F')
    
    # If there are no attributes, build the node with +ve or -ve based on their count
    if len(attributes) == 0:
        pos = 0
        neg = 0
        pos = (Xtrain['target'] == 1).sum()
        neg = (Xtrain['target'] == 0).sum()
        if pos >= neg:
            leaf_nodes += 1 # Increment no.of leaf_nodes
            return TreeNode()            
        else:
            leaf_nodes += 1 # Increment no.of leaf_nodes
            return TreeNode('F') 
    
    # Get the best attribute to split on based on the Maximum Entropy Gain
    best_attribute = find_attribute_to_split_on(Xtrain, attributes)    
    attributes.remove(best_attribute)
    node = None

    # Calculate the p_value for the chosen attribute
    p = chi2_func(Xtrain, best_attribute) 

    # Build the node only if the obtained value if less than p_val  
    if p >= pval:                     
        return None
    else:        
        node = TreeNode(best_attribute + 1)
        internal_nodes += 1 # Increment internal nodes

        i = 1
        pos_miss = -1
        neg_miss = -1

        while i < 6:

            # Build the child nodes for the chosen attribute node
            sample_space_func(Xtrain, best_attribute)
            if i in Xtrain[best_attribute].unique():
                Xtrain_subset = Xtrain.loc[Xtrain[best_attribute] == i]

                if Xtrain_subset.empty:
                    pos = (Xtrain['target'] == 1).sum()
                    neg = (Xtrain['target'] == 0).sum()
                    if pos < neg:
                        leaf_nodes += 1 # Increment no.of leaf_nodes
                        node.nodes[i - 1] = TreeNode('F')
                    else:
                        leaf_nodes += 1 # Increment no.of leaf_nodes
                        node.nodes[i - 1]= TreeNode()
                else:       
                    is_node = create_tree(Xtrain_subset, attributes, pval)

                    if is_node:
                        node.nodes[i - 1] = is_node
                    else:
                        pos = (Xtrain_subset['target'] == 1).sum()
                        neg = (Xtrain_subset['target'] == 0).sum()
                        if pos < neg:
                            leaf_nodes += 1 # Increment no.of leaf_nodes
                            node.nodes[i - 1] = TreeNode('F')
                        else:
                            leaf_nodes += 1 # Increment no.of leaf_nodes
                            node.nodes[i - 1]= TreeNode()
            else:
                if pos_miss == -1 and neg_miss == -1:
                    pos_miss = (Xtrain['target'] == 1).sum()
                    neg_miss = (Xtrain['target'] == 0).sum()
                if pos_miss < neg_miss:
                    leaf_nodes += 1 # Increment no.of leaf_nodes
                    node.nodes[i - 1] = TreeNode('F')
                else:
                    leaf_nodes += 1 # Increment no.of leaf_nodes
                    node.nodes[i - 1] = TreeNode()

            i += 1
    return node    

# traverse the tree and return the best possible value for the test data        
def get_class_of_datapoint(root, datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return get_class_of_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)  

# Parse the input Command arguments given by the User to obtain the Train,Test,Output filenames    
parser = argparse.ArgumentParser()
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_label.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

# Load the Train and Test Data from the files specified by the User
Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

Xtrain['target'] = Ytrain[0] # append labels column to the features set
attribute_count = Xtrain.shape[1] - 1
attributes = [i for i in range(attribute_count)]

print("Training...")

# Build the Tree using the Training Data
root = create_tree(Xtrain, attributes, float(pval))   
root.save_tree(tree_name)  

print("Testing...")
Ypredict = []

# Get the best possible output values for the Test Data
for i in range(0,len(Xtest)):
    Ypredict.append([get_class_of_datapoint(root, Xtest.loc[i])])

# Write the Predicted Test Output values to a file with the name as specified by User
with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)
print("Output files generated")

print("No.of Internal Nodes: ", internal_nodes)
print("No.of Leaf Nodes: ", leaf_nodes)
print("Total No.of Nodes: ", internal_nodes + leaf_nodes)