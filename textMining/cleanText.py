import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import json
import os
import pickle



#########################################################

# Preprocess news before LDA analysis

#-----------------------------------------------------

def preprocessText(text,remove_number,convert_to_noun,convert_to_verb):

    '''
    Precess text before doing IDA analysis.

    Args: 
    text(str): a single news

    Returns:
    filtered(list): a list of a news
    '''
    # Convert words into lowercase. For example, "Home" is converted to "home".  
    text = text.lower()
                                   

    # Remove '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' from text
    for c in string.punctuation:                   
        text = text.replace(c, ' ')

    if remove_number:
        # Remove numbers 
        number_list = ['0','1','2','3','4','5','6','7','8','9']
        for i in number_list:
            text = text.replace(i," ")

    # Split sentence in a list of words
    wordLst = nltk.word_tokenize(text)              

    # Remove stopwords from text. 
    # Stopwords are words like "me","I","hasn't" and so on.
    filtered = [w for w in wordLst if w not in stopwords.words('english')]    

    # Remove words with letters less than 3
    filtered = [w for w in filtered if len(w) < 3]
    


    if convert_to_noun:
        filtered = [nltk.stem.WordNetLemmatizer().lemmatize(w, 'n') for w in filtered]
        filtered = [w for w in filtered if len(w) < 3]
        filtered = [w for w in wordLst if w not in stopwords.words('english')] 


    if convert_to_verb:
        filtered = [nltk.stem.WordNetLemmatizer().lemmatize(w, 'v') for w in filtered]
        filtered = [w for w in filtered if len(w) < 3]
        filtered = [w for w in wordLst if w not in stopwords.words('english')] 


    return " ".join(filtered)



#-----------------------------------------------------

def combineNewsViaPickle(pickle_preprocess_news_name):
    '''
    This function is to combine all news in the sample pickle file by multiple times of loading the file.

    Args:
    pickle_preprocess_news_name(str): The full filename of pickle file that contains all preprocessed news.

    Returns:
    docLst_all[list]: The list that combines all news in the same pickle file.
                      The length of "docLst_all" is the total number of all news.
                      
    '''
    docLst_all = []

    f = open(pickle_preprocess_news_name,'rb')

    while True:
        try:
            docLst = pickle.load(f)
            for line in docLst:
                if line != '':
                    docLst_all.append(line)

        except:
            break
    return docLst_all



#----------------------------------------------------

def preprocessNewsViaPickle(pickle_raw_news_name,pickle_output_name,remove_number=True,convert_to_noun=True,convert_to_verb=False):
    '''
    This function is to preprocess news in one pickle file and save the preprocess news(lists of words) in another pickle file.

    Args:
    pickle_raw_news_name(str): The full filename of pickle file that contains all raw news.
    pickle_output_name(str): The full filename of pickle file where preprocessed news is saved in.
    remove_number(bool): If this is set as "True", all numbers and datetimes would be removed from text.
    convert_to_noun(bool): If this is set as "True", words would be converted to Nouns.
    convert_to_verb(bool): If this is set as "True", words would be converted to Verbs.

    ** Attention: "convert_to_noun" and "convert_to_verb" can not be both set as "True" in a time.

    Returns: All preprocessed news are saved in the pickle file with name as "pickle_output_name".
     
    '''
    
    f_input = open(pickle_raw_news_name,'rb')

    f_output = open(pickle_output_name,'wb')
    
    n=0

    while True:
        try:
            contents = pickle.load(f_input)
            
        except:
            break
            
        data = contents['data']

        docLst = []


        for i in range(len(data)):    
            desc = data[i]['teaser']
            docLst.append(preprocessText(desc,remove_number,convert_to_noun,convert_to_verb))
        pickle.dump(docLst,f_output)

    f_input.close()
    f_output.close()

    print('News preprocess done. Output pickle is '+pickle_output_name)

