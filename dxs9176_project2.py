#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:26:58 2019

@author: dikshasharma
"""

from collections import Counter
import os, math, random, re
from functools import reduce
import numpy as np

stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
 'each', 'few', 'for', 'from', 'further', 
 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 
 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
 "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't", 
 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th']

#reading dataset through path directory
doc_list = {}
test_data = {}
train_data = {}

directory_for_data = os.listdir(os.getcwd() + '/20_newsgroups')

for folders in directory_for_data:
    paths_folder = os.getcwd() + '/20_newsgroups/' + folders + '/'
    doc_path_text = os.listdir(os.getcwd() + '/20_newsgroups/' + folders)
    doc_list[folders] = doc_path_text
    shuffled_list_data = list(range(0, len(doc_path_text)))
    random.shuffle(shuffled_list_data)

#splitting dataset into 50-50
    a=shuffled_list_data[int((len(doc_path_text) / 2)):]
    c=shuffled_list_data[:int((len(doc_path_text) / 2))]
    train_data[folders] = list(map(lambda data1: doc_path_text[data1], a))
    test_data[folders] = list(map(lambda data1: paths_folder + doc_path_text[data1],c ))
#train_data1=pd.DataFrame.from_dict(train_data,orient="index")
    
    
##removing lines at the top of the document
def remove_metadata(lines):
    new_lines=[]
    start=0
    for i in range(len(lines)):

        if(lines[i] == '\n'):
            start = i+1
            break
        new_lines = lines[start:]
    return new_lines


#prerocessing the words after reading them
def vocabulary_extraction(file):
    with open(file, 'rb') as word:
        line_by_line_str = word.read().lower()
        s=remove_metadata(line_by_line_str)
        q=re.findall(rb"[\w']+", s)
        stop_words_removal = np.array([word for word in q if not word in stopwords])      
        #removing any digits
        removing_numbers = np.array([word for word in stop_words_removal if not word.isdigit()])        
        #removing words of length 1
        words_length_1_removal = np.array([word for word in removing_numbers if not len(word) == 1])       
        #removing words of length 2
        words_length_2_removal = np.array([word for word in words_length_1_removal if len(word) > 2])      
        #removing words if it is not a string
        string_removal_not = np.array([str for str in words_length_2_removal if str])      
        #removing words if it is alphanumeric
        alpnum_removed_words = np.array([word for word in string_removal_not if word.isalnum()])
        
        return alpnum_removed_words
        
# Unique Word Counting in order to get the most frequently used words.
list_path=list(map(lambda x: len(doc_list[x]), directory_for_data))
word_count_set = dict(zip(directory_for_data,list_path ))
count_training_set_words = {}
count = Counter()
sum_of_count_train_data = {}

for key in directory_for_data:
    print(key)
    cwd = os.getcwd() + '/20_newsgroups/' + key + '/'
    cnt = Counter()
    for fi in train_data[key]:
        cnt = cnt + Counter(vocabulary_extraction(cwd + str(fi)))
    count = count + cnt
    count_training_set_words[key] = dict(cnt)
    sum_of_count_train_data[key] = sum(count_training_set_words[key].values())
count = len(dict(count).keys())
print(count_training_set_words)

#-------Naive Bayes Implementation----------

# probability of the class words by Laplace smoothing
def probability_word_laplace_smoothing(hash, word, denominator_val):
    word = word.lower()
    if word in hash:
        return math.log(hash[word] + 1.0) / (denominator_val+1)
    else:
        return math.log(1.0 / denominator_val)

#  combining word probabilities for document categories 
def totalProbability(file, hash, denominator_val):
    li = list(map(lambda x: probability_word_laplace_smoothing(hash, x, denominator_val), vocabulary_extraction(file)))
    prob=reduce(lambda x, y: x + y, li)
    return prob

# training the naive bayes model on words
def train_classify(file, hashing, traningSum, counter):
    print("path for the document-",file)
    keys = traningSum.keys()
    probability = list(map(lambda x: totalProbability(file, hashing[x], sum_of_count_train_data[x] + counter), keys))
    probability = list(map(lambda x: x - max(probability), probability))
    denominator_val = sum(list(map(lambda x: math.exp(x), probability)))
    probability = list(map(lambda x: math.exp(x) / denominator_val, probability))
    print("probability-",max(probability))
    print("------------------------------------------------------------------------------------------")
    maximumIndex = [index for index in range(len(probability)) if probability[index] == max(probability)]
    if (len(maximumIndex) > 1):
        print('Cannot classify as it has the same probability')
    return list(keys)[maximumIndex[0]]


# Classifying test data using the trained model
count=0
print(list(test_data.keys()))
length = len(test_data.keys())
for i in range(0, length):
    length_file = len(test_data[list(test_data.keys())[i]])
    for j in range(0, length_file):
        print(" Test data train_classify calculation-")
        final_op=train_classify(test_data[list(test_data.keys())[i]][j], count_training_set_words, sum_of_count_train_data, count)
        print("Folder in which the test data lie-",final_op)
        
#accuracy calculation
count1=0
for key in directory_for_data:
    if final_op==key:
        count1+=1
print("Error",count1/length)
    
        
            
     

    
            
      
    