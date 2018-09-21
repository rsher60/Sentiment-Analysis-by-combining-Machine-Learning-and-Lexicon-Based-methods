#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:57:59 2018

@author: MyReservoir
"""

#ALDA Project- NLP
 
import random
import os
import numpy as np
import pandas as pd
import treepredict as dt

os.getcwd()
os.chdir('/Users/MyReservoir/Desktop/CSC 522 PROJECT/')
os.getcwd()

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from pandas import DataFrame


no_of_records=4000
no_of_trials=10
# ----- FUNCTIONS ------

# Data Preprocessing functions

def nltk_process(sample_text):
    #reading an input text file
    #input_file = open('/Users/jagadeesh/PycharmProjects/ALDA/testdoc.txt','r')
    #sample_text = input_file.read()
    #input_file.close()


    #word tokenizer, but it includes the punctuation
    #tokenized = word_tokenize(sample_text)
    #tokenizing words using RegexpTokenizer of nltk, which by default removes all punctuation.
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(sample_text)
    length_input = len(tokenized) #used later in for loop which splits pos_tags

    # print(tokenized)
    #converted each word to its root word using WordNetLemmatizer of nltk.
    lemma = WordNetLemmatizer()
    lemma_words = map(lemma.lemmatize, tokenized)
    pos_tagged = []

    #function for assigning pos_tags
    def pos_tagging():
        try:
            for i in lemma_words:
                words = nltk.word_tokenize(i)
                for j in words:
                    tagged = nltk.pos_tag(words)
                    pos_tagged.append(tagged)
        except Exception as e:
            print(str(e))
    
    #function call for performing pos_tags
    pos_tagging()
    #print(pos_tagged)
    #splitting the pos_tags into two seperate lists for further processing
    list_df = DataFrame.from_records(pos_tagged)
    words = []
    pos_final = []
    for i in range(0, length_input):
        for j in range(0,2):
            if j == 0:
                words.append(list_df[0][i][j])
            if j == 1:
                pos_final.append(list_df[0][i][j])
    #print(words)
    #print(pos_final)

    #converted as table if required for further processing
    pos_table = np.column_stack((words, pos_final))
    #pos_table=pd.DataFrame(words,pos_final)
    return pos_table

def ads(st):
    merge = pd.merge(st, pd.DataFrame(dic), how='left', left_on='Word', right_on='Token', sort=False)
    merge = merge.fillna(0)
    merge2 = pd.merge(merge, pd.DataFrame(intense), how='left', left_on='Word', right_on='Token', sort=False)
    merge2 = merge2.fillna(0)
    
    for i in range(len(merge2.index)-1):
        merge2['Polarity_x'][i+1]=merge2['Polarity_x'][i+1]*(1+merge2['Polarity_y'][i])
        
        merge3 = pd.merge(merge2, pd.DataFrame(bucket), how='left', left_on='POS', right_on='POS', sort=False)
        merge4 = pd.DataFrame(merge3.groupby(by = ['Bucket']).agg({'Polarity_x' : 'sum'}).transpose())

        return merge4

#Classification code
#Define function to split dataset with ratio
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = dataset.values.tolist()
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


# Data Preprocessing 
data = pd.read_csv('/Users/MyReservoir/Desktop/CSC 522 PROJECT/tweetylabel.csv',encoding ='utf=8')
dic = pd.read_csv('/Users/MyReservoir/Desktop/CSC 522 PROJECT/dic.csv',encoding ='utf=8')
intense = pd.read_csv('/Users/MyReservoir/Desktop/CSC 522 PROJECT/intense.csv',encoding ='utf=8')
bucket = pd.read_csv('/Users/MyReservoir/Desktop/CSC 522 PROJECT/bucket.csv',encoding ='utf=8')

data=data[:no_of_records]
data.columns = ['Tweet','Sentiment']

output_nltk = []
for i in range(len(data)):
    sample_text = data['Tweet'][i]
    output = nltk_process(sample_text)
    output_nltk.append(output)

ads_df = pd.DataFrame({'Adjective':[],'Adverb':[],'Noun':[],'Verb':[]})

for i in range(len(output_nltk)):
    nltkout_temp = pd.DataFrame(output_nltk[i])
    #nltkout_temp = pd.DataFrame(sample_tweet)
    nltkout_temp.columns = ['Word','POS']
    nltkout_temp['Word'] = pd.DataFrame(nltkout_temp['Word'].str.lower())
    ads_df = ads_df.append(ads(nltkout_temp))
    ads_df = ads_df.fillna(0)

ads_df['index'] = range(1, len(ads_df) + 1)
data['index'] = range(1, len(data) + 1)
#data['label'] = data.apply(f,axis=1)
ads_df_final = pd.merge(ads_df, pd.DataFrame(data), how='left', left_on='index', right_on='index', sort=False)
dataset = ads_df_final[['Adjective','Adverb','Noun','Verb','Sentiment']]



############# Splitting the Dataset into Testing and Training Sets ##############   
final_acc=0.0

for i in range(no_of_trials):
    splitRatio = 0.7
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    #print(trainingSet)
    #    print(type(trainingSet))
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))
    
    ############# Model Building ##############   
    b = dt.buildtree(trainingSet)
    dt.drawtree(b,jpeg='treeview.jpg')
    
    #print("original_testset=",testSet)
    ############# Preparing Testing DataSet ##############   
    testlabels=[]
    for i in range(len(testSet)):
        label=testSet[i].pop(-1)
        testlabels.append(label)
    
    #print("testSet=",testSet)
    #print("testlabels=",testlabels)
    ############# Classification of Test Records ##############   
    number = 0
    for i in range(len(testSet)):
        #print("\ntest_data",testSet[i])
        a = dt.classify(testSet[i], b)
        #print("a=",a)
        max=0
        best=""
        for key in a.keys():
            if a[key]>max:
                max=a[key]
                best=key
        #print("best=",best)
        #print("label=",testlabels[i])
        if(best == testlabels[i]):
            number = number + 1
           
    ############# Accuracy Calculations ##############   
    accuracy = (number/len(testSet))* 100
    final_acc+=accuracy

final_acc=final_acc/no_of_trials
print('Accuracy: {0}%'.format(final_acc))