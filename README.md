# Generating_Random_Sentences_with_N-gram_Language_Model
NLP, N-gram model, Markov assumption, text-processing

1. Project objective

    - Build an N-gram language model from an arbitrary number of plain text files. 
    - Generate a given number of sentences based on that N-gram model


2. N-gram model

    - Separate punctuation marks from text and treat them as tokens. numeric data is treated as tokens
    - identify sentence boundaries, and n-grams should *not* cross these boundaries. 



    Please follow the following example to input the command.

   '>python ngram.py 3 10 pg2554.txt pg2600.txt pg1399.txt'
    
        - 3 means Trigrams 
        - 10 means 10 random sentences will be generated 
        - The last argument is the book name, or a book list. Please put the book/books in the same path with the script
