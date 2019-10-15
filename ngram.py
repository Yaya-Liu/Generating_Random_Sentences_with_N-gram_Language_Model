# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:52:21 2019

@author: Yaya Liu
"""
import sys
import random
import re
import nltk
import logging
import time

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.util import ngrams
import string
punctuations = list(string.punctuation)


log = "ngram-log.txt"   # create a log file

logging.basicConfig(filename=log,level = logging.DEBUG, format = '%(message)s')
logging.info('%script ngram-log.txt')   # start logging
 


# print out the project description
def project_description():
    print()
    print("This program generates random sentences based on an Ngram model.")
    print()
    print("Author: Yaya Liu")
    print()
    print("Please follow the following example to input the command.")
    print()
    print("'>python ngram.py 3 10 pg2554.txt pg2600.txt pg1399.txt'")
    print("     - 3 means Trigrams \n", "    - 10 means 10 random sentences will be generated \n", "    - The last argument is the book name, or a book list. Please put the book/books in the same path with the script")
    print()

# get and return inputs: n, m, and file names
def get_inputs(): 
    input_len = len(sys.argv) - 1    # subtract ngram.py
    
    if input_len < 3:        # check whether there are less than 3 arguments, if not, exit the script
        print("Please enter at least 3 arguments.")
        sys.exit()
      
    N = int(sys.argv[1])   # get the first argument: N-gram 
    M = int(sys.argv[2])   # get the second argument: the number of sentences
    file_names = [str(x) for x in sys.argv[3:]]  # get the list of the books
    
    if N <= 0:     # check the N for N-gram
        print("The first argument should be greater than 0.")
        sys.exit()
    elif M <= 0:  # check whether the number of sentences is equal and greater than 1
        print("The second argument should be greater than 0.")
        sys.exit()
    
    print("Command line settings: ")
    print("Ngram:", N, ", Number of sentences:", M, ", File names:", file_names, end = "\n")
    print()
    
    return N, M, file_names


################################ Unigram ##################################################
# read files, create unigram and calcuate the raw frequency of unigram 
def get_unigram(file_names):
    my_tokens = []
    
    # get the nested list of N-gram and (N-1)gram
    for name in file_names: 
        with open (name, errors = 'ignore') as fin:
            sent_tokens = sent_tokenize(fin.read())  # tokenize the text to sentences      
            for sent in sent_tokens:
                sent = sent.lower()                  # convert all words to lower case
                
                sent = sent + " EOF"  # set the sentence's boundary with "EOF"               
                tokens = re.findall(r"[\w]+|[^\s\w]", sent) # use re.findall to split the sentence, without removing the separators          
                                
                for token in tokens:                    
                    my_tokens.append(token)
                                       
    # If the length of the whole corpus is less than a token, then print the message and exit 
    if len(my_tokens) < 1:          
        print("Please include more texts.")
        sys.exit()
    
    my_unigram_fdist = nltk.FreqDist(my_tokens)    # calculate raw frequency of unigram          
    return my_unigram_fdist, my_tokens        

# generate sentences based on unigram
def gen_sent_unigram(M, my_unigram_fdist, my_tokens):
    
    uni_relfreq_list = []
    
    temp = 0   
    
    # build a nested list to store [(token, cumulative relative frequency),...]
    for k,v in my_unigram_fdist.items():
        temp = temp + v/len(my_tokens)
        uni_relfreq_list.append([k, temp])      

    for m in range(0, M):        
        sentence, word = '', ''
        notEnd = True
                   
        while(notEnd):              
            prob = random.uniform(0, 1)  # assign a random number between 0 and 1
            
            # use binary search to find where "prob" falls
            left = 0
            right = len(uni_relfreq_list) - 1  
              
            while (right - left > 1):
                mid = int((left + right) / 2)
                
                if uni_relfreq_list[mid][1] > prob:
                    right = mid
                elif uni_relfreq_list[mid][1] < prob:
                    left = mid
                else: 
                    word = uni_relfreq_list[mid][0]
            
            if uni_relfreq_list[left][1] > prob:
                word = uni_relfreq_list[left][0]
            elif uni_relfreq_list[left][1] < prob:
                word = uni_relfreq_list[right][0]
            else: 
                word = uni_relfreq_list[left][0]
            
            if word in string.punctuation and len(word_tokenize(sentence)) == 0: # if the first word is a punctuation , then skip it
                continue
            elif word == 'EOF' and len(word_tokenize(sentence)) <= 20:    # if the word is 'EOF', but the sentence is less than 20 words, then skip it          
                continue
            else:
                sentence = sentence + " " + word
                
                # if the selecte word == "end", then end the setence creation
                if word == 'EOF':  # make sure the sentence has at least 1 word. 
                    notEnd = False
        
        sentence = re.sub(r'EOF', '', sentence)   # remove "STA" and "EOF" from the sentence  
        print("Sentence", m + 1, ": ", sentence)
        print()   

################################ N-gram (N >= 2) ##################################################
# use zip function to create N-gram based on tokens
def create_ngrams(N, tokens):
    ngrams_list = zip(*[tokens[i:] for i in range(N)])
    return [" ".join(ngram) for ngram in ngrams_list]
              
# read files, create N-gram and (N-1)gram and calcuate the raw frequency of N-gram and (N-1)gram 
def get_ngrams(N, file_names):
    my_ngrams, n_1_grams, my_tokens = [], [], []
    my_tokens = []
    
    # get the nested list of N-gram and (N-1)gram
    for name in file_names: 
        with open (name, errors = 'ignore') as fin:
            sent_tokens = sent_tokenize(fin.read())  # tokenize the text to sentences      
            for sent in sent_tokens:
                sent = sent.lower()                  # convert all words to lower case
                
                new_sent = "STA " + sent + " EOF"  # set the sentence's boundary with "STA" and "EOF"
                
                tokens = re.findall(r"[\w]+|[^\s\w]", new_sent) # use re.findall to split the sentence, without removing the separators          
                #print("tokens*********", tokens)
                                
                # If the length of a sentence is greater and equal than ntokens.  
                if len(tokens) - 2 >= N:       # -2 means subtract "STA" and "EOF"        
                    for token in tokens:                    
                        my_tokens.append(token)                   
                else:
                    continue
                      
    
    # If the length of the whole corpus is less than ntokens, then print the message and exit 
    if len(my_tokens) - 2 < N:          
        print("Please include more texts.")
        sys.exit()
    
    my_ngrams = create_ngrams(N, my_tokens)             # create the list of Ngram
    n_1_grams = create_ngrams(N-1, my_tokens[0:-1])     # create the list of (N-1)gram
    
    n_1_grams_fdist = nltk.FreqDist(n_1_grams)     # calculate raw frequency of (N-1)gram
    
    my_ngrams_fdist = nltk.FreqDist(my_ngrams)    # calculate raw frequency of N-gram 
               
    return my_ngrams_fdist, n_1_grams_fdist, my_tokens   

 
# get the given word/words
def get_condi_token(i, N, my_tokens):
    temp = []
    for j in range(1, N):
        temp.append(my_tokens[i-j])
        
    temp.reverse()
    condi_token = ' '.join(map(str, temp))
    return condi_token
      

# build relative frequency dictionary
def cal_relfreq(my_ngrams_fdist, n_1_grams_fdist, my_tokens, N):
    
    my_ngrams_rel_dict, n_1_grams_rel_dict = {}, {}
    
    for k, v in my_ngrams_fdist.items():    # create the relative frequency dictionary for N-gram
        my_ngrams_rel_dict[k] = v/len(my_ngrams_fdist)
   
    for k, v in n_1_grams_fdist.items():    # create the relative frequency dictionary for (N-1)-gram
        n_1_grams_rel_dict[k] = v/len(n_1_grams_fdist)
    
    relfreq_dict = {} 
    
    # create a dictionary, key is a tuple(prediction word, given word/words), value is (N-gram relative frequency/N-1 gram relative frequency)
    for i in range(N-1, len(my_tokens)): # iterate all the tokens
        condi_token = get_condi_token(i, N, my_tokens) 
        #print(i, condi_token)
        
        relfreq_dict[(my_tokens[i], condi_token)] =  my_ngrams_rel_dict[condi_token + " " + my_tokens[i]]/n_1_grams_rel_dict[condi_token]

#    print("\n")
#    print("relfreq_dictionary: ")
#    for k,v in relfreq_dict.items():
#        print(k, ":" , v, end = "   ")        
        
    return relfreq_dict

# pick up the next(prediction) word for N-gram
def pickup_word(next_keys, relfreq_dict):
    total_prob, temp = 0, 0
    word = ''
    select_list = []
    
    # calculate the sum of the probability
    for key in next_keys:
        total_prob += relfreq_dict[key]
        
    # normalize and calculate the cumulative relative prbability. 
    # create a list = [[prediction, given, cumulative relative probability] ... ]
    for key in next_keys:
        temp = temp + relfreq_dict[key]/total_prob
        select_list.append([key[0], key[1], temp])
                       
    prob = random.uniform(0, 1)  # assign a random number between 0 and 1
    
    # use binary search to find where "prob" falls
    left = 0
    right = len(select_list) - 1  
      
    while (right - left > 1):
        mid = int((left + right) / 2)
        
        if select_list[mid][2] > prob:
            right = mid
        elif select_list[mid][2] < prob:
            left = mid
        else: 
            word = select_list[mid][0]
    
    if select_list[left][2] > prob:
        word = select_list[left][0]
    elif select_list[left][2] < prob:
        word = select_list[right][0]
    else: 
        word = select_list[left][0]    

    return word

# generate sentences for N-gram
def gen_sent(N, M, relfreq_dict):    
    for m in range(0, M):
        start_keys = []
        
        # find the keys which second element starts with "STA" and the relative frequency is not 0
        start_keys = [key for key in relfreq_dict.keys() if 'STA' == key[1].split(" ")[0] and relfreq_dict[key] != 0]
        #print("start_keys: ", start_keys)
                    
        # random choose the second element of the key. (the second element of the key is the given word/words)  
        first_word = random.choice(start_keys)[1]  
        #print()
        #print("sentence's first_word: ", first_word)
        
        sentence = first_word   # assign the first word to the sentence
        
        NumofPick = 1           # how many words have been picked up
        
        notEnd = True
        
        while(notEnd):           
            if NumofPick == 1:     # just picked up the 1st word
                
                # find the keys which second element is the first word and the relative frequency is not 0
                next_keys = [key for key in relfreq_dict.keys() if first_word == key[1]]
                NumofPick = 2
            
            elif NumofPick > 1:    # already found more than 1 word  
                tokens = re.findall(r"[\w]+|[^\s\w]", sentence)  # split the partially created sentence
                given = tokens[-(N-1):]            # find the tokens of the given word/words
                given = ' '.join(map(str, given))  # create given word/words 
                #print("given tokens: ", given)  
                
                # find the keys which second element is the given word/words and the relative frequency is not 0
                next_keys = [key for key in relfreq_dict.keys() if given == key[1]]
                NumofPick += 1 
            
            #print("next_keys: ", next_keys)
            
            # pick up the next word from "next_keys[0]"              
            next_word = pickup_word(next_keys, relfreq_dict)
            #print("next_word: ", next_word)   
            
            sentence = sentence + " " + next_word  # add the selected word to the sentence
            
            # if the selecte word includes "EOF", then end the setence creation
            if next_word == 'EOF': 
                notEnd = False
        
        sentence = re.sub(r'STA', '', sentence)   # remove "STA" from the sentence  
        sentence = re.sub(r'EOF', '', sentence)   # remove "EOF" from the sentence
        print("Sentence", m + 1, ": ", sentence)
        print()
                                 
              
def main():
    project_description()
    
    start_time = time.localtime()   # get start time
    
    N, M, file_names = get_inputs()
    
    if N > 1:           
        my_ngrams_fdist, n_1_grams_fdist, my_tokens = get_ngrams(N, file_names)      
        relfreq_dict = cal_relfreq(my_ngrams_fdist, n_1_grams_fdist, my_tokens, N)                
        gen_sent(N, M, relfreq_dict) 
        
    elif N == 1:
        my_unigram_fdist, my_tokens = get_unigram(file_names)                
        gen_sent_unigram(M, my_unigram_fdist, my_tokens)
        
    ### log processing    
    process_time = time.mktime(time.localtime())-time.mktime(start_time)  # get process time in seconds
    
    logging.info("%s minutes, python ngram.py %d %d %s", round(process_time/60, 5), N, M, file_names)   # log process time in minutes
    
    logging.info('%exit')   # exit logging
    ### log end  
    
if __name__ == "__main__":
    main()