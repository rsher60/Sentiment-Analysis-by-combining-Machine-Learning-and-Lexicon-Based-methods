# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:30:59 2018

@author: Riddhiman Sherlekar & Harsh Mehta
"""


import pandas as pd             #importing pandas library to manipulate data frames
import numpy as np              # importing numpy library for mathematical calculation
import operator                 #
data = pd.read_csv("C:\\Users\\rsher\\Desktop\\features_update_2.csv")      #reading the data into a data frame


class classifier:
    def __init__(self, true=None, false=None, column=-1, split=None, result=dict()):
        self.true=true    #it is used to identify the true decision nodes
        self.false=false #it is used to identify the false decision nodes
        self.column=column #the column index of criteria that is being tested
        self.split=split #it assigns the split point based on entropy
        self.result=result #it stores the result in the form of an dictionary
        


def key_labels(data):            #This function is used to return the unique class labels from the data                        
    keys = []
    for i in range(0, len(data)):   #For loop is used to store unique class/keys from dataset 
        if data.iloc[i,len(data.columns)-1] in keys:
            continue
        else:
            keys.append(data.iloc[i,len(data.columns)-1])
    return keys

keys = key_labels(data) 

def entropy_cal(data, keys):    #This function is used to return the entropy of parent node / data and the count of class 
    count = [0]*len(keys)
    ent = [0]*len(keys)
    lgth = len(data)
    for i in range(0,len(data)):    #A for loop is used to calculate the count of each class in a dataset
        for j in range(0,len(keys)):
            if data.iloc[i,len(data.columns)-1] == keys[j]:
                count[j] += 1
        for j in range(0,len(keys)):    #A for loop is used to calculate the entropy of data
            if count[j] != 0:
                ent[j] = (count[j]/float(lgth))*np.log2(count[j]/float(lgth))   #Numpy Library is used for calculating the Log2
    ent = [-x for x in ent]
    entropy = sum(ent)
    return entropy, count

def entropy_data(data, keys):   #This fucntion is used to calculate entropy of each attribute at every split point and then store the attribute, best split point and entropy
    len_keys = len(keys)
    ent_dict = {}
    split_dict = {}
    #print(data)
    if len(data.columns) > 1:   #If loop is executed if the number of columns in dataset is greater than 1

        for i in range(0,len(data.columns)-1):  #A for loop is executed to carry out entropy calculations for each attribute 
            if len(data[data.columns[i]].unique()) > 1: #If the number of unique values in an attribute is greater than one, then the if loop is executed
                entropy_min = np.log2(len(keys)) # Initially the minimum value of entropy is set as the maximum possible entropy value, i.e. log(n) where n is the number of unique classes
                test = data[data.columns[[i,len(data.columns)-1]]]  #A dataset is created having the selected attribute and its label values to find the split point
                max_value = test.iloc[:,0].max()
                min_value = test.iloc[:,0].min()
                y = min_value       #the initial value of split point is set as the minimum value of the selected attribute
                while (y < max_value):  #a while loop is executed untill the split point value reaches the maximum value of that attribute
                    left = []   #an empty list is created to store the labels of the dataset with value <= split point
                    right = []  #an empty list is created to store the labels of the dataset with value > the split point
                    for x in range(0, len(test)):   #a for loop is created to append the attribute values in the "left" and "right" lists
                        if test.iloc[x,0] <= y:
                            left.append(test.iloc[x,1])
                        else:
                            right.append(test.iloc[x,1])
    
                    
                    split_pair = [left, right]
                    ent_pair = [0,0]
                    for sp_val in range(0,2):       #A for loop is created to calculate the entropy of the left split and right split
                        count = [0]*len_keys
                        prop = [0]*len_keys
                        for sp_ct in range(0,len(split_pair[sp_val])):
                            lgth = len(split_pair[sp_val])
                            for j in range(0,len_keys):
                                if split_pair[sp_val][sp_ct] == keys[j]:
                                    count[j] += 1 
                        for j in range(0, len_keys):
                            if count[j] != 0:
                                prop[j] = (count[j]/float(lgth))*np.log2(count[j]/float(lgth))
                        ent_pair[sp_val] = sum(prop)
                            
                    ent_pair = [-x for x in ent_pair]
                    entropy = (ent_pair[0]*len(left)/float(len(test))) + (ent_pair[1]*len(right)/float(len(test)))  #The resultant entropy of the split is stored in "entropy"
                    if entropy < entropy_min:   #if the entropy of the split is less than the minimum entropy, then the if loop is executed which will overwrite the minimum entropy and best split point
                        splitvalue = y
                        entropy_min = entropy
                    y = y + 1
                ent_dict[test.columns[0]]  = entropy_min    #the minimum entropy of each attribute is appened in the dictionary ent_dict having the attribute as the key
                split_dict[test.columns[0]] = splitvalue    #attribute is stored as a key and the split point is stored as its value in the dictionary split_dict
    return ent_dict, split_dict


def parent_node(dictionary, entropy):
    gain = {}
    for k,v in dictionary.items():
        gain[k] = entropy - v  
    attribute = max(gain, key=gain.get)
    return attribute, gain[attribute]

def split_data(data, attribute, split_dict):
    left = data[data[attribute] <= split_dict[attribute]]
    right = data[data[attribute] > split_dict[attribute]]
    return left, right


def unique_values(data1, data2):
    left_count = 0
    right_count = 0
    for i in (0,len(data1.columns)-1):
        left_idx = len(data1[data1.columns[i]].unique())
        left_count = left_count + left_idx
    for i in (0,len(data2.columns)-1):
        right_idx = len(data2[data2.columns[i]].unique())
        right_count = right_count + right_idx
    return left_count, right_count


        
def complete_tree(data, keys):  #This function is used to store the rules/decisions in an object through a class called "classifier"
    entropy, count = entropy_cal(data,keys)     #The entropy of parent node is stored in the variable "entropy"
    ent_dict, split_dict = entropy_data(data, keys) 
    attribute, gain = parent_node(ent_dict, entropy)    #the attribute and gain is stored using the function parent_node
    if gain > 0:    #An if loop is executed when the gain will be positive
        left, right = split_data(data, attribute, split_dict)   #split_data function is used to split the dataset into two based on the attribute and split value
        left_count, right_count = unique_values(left, right)    #the count of unique values from each attribute is calculated
        if left_count > len(left.columns) - 1:  #if each attribute has only one value, then the if loop is not executed
            true = complete_tree(left, keys)
        if right_count > len(right.columns) - 1:
            false = complete_tree(right, keys)
        if left_count > len(left.columns) - 1 and right_count > len(right.columns) - 1: 
            return classifier(true=true, false=false, column = attribute, split = split_dict[attribute]) #the attribute and split value is stored in the class
        else:
            return classifier(result = dict(zip(keys, count)))  #the count of each class at the leaf node is stored as results in a class
    else:
        return classifier(result = dict(zip(keys, count)))

a=complete_tree(data,keys)

keys = key_labels(data)

a = complete_tree(data,keys)


###################################################################
                   #BUILDING THE MODEL#

#################################################################
                    
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.25)

test22 = test.iloc[:, :-1]

test_data = test22.to_dict('records')

#test['result']=""




#type(test_data)


def traverse(a, each_row):
    if(a.column == -1):
        return max(a.result.iteritems(), key=operator.itemgetter(1))[0]
    if(each_row[a.column] < a.split ):
        return traverse(a.true, each_row)
    else:
        return traverse(a.false, each_row)

for each_row in test_data:
    label_value = traverse(a, each_row)
    each_row['Label'] = label_value
    
test_data2 = pd.DataFrame(test_data)

labels_test_data= test_data2[test_data2.columns[[2]]]

labels_of_original = test[test.columns[[-1]]]

count=0
for i in range(0,len(labels_test_data)):
    if labels_test_data.iloc[i,0] == labels_of_original.iloc[i,0]:
        count = count + 1
count

































    
    
