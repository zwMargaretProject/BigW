import numpy as np
import json
import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import matplotlib.pyplot as plt
from time import time

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
        except:
            break
        for line in docLst:
            if line != '':
                docLst_all.append(line)
    f.close()
    return docLst_all

#----------------------------------------------------

def getTopWordsList(model, feature_names, n_top_words):
    '''
    Show results of LDA.

    '''
    top_words_list = []

    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        top_words_list.append(top_words)
    return top_words_list


#---------------------------------------------------------
def print_top_words(model, feature_names, n_top_words):
    '''
    Show results of LDA.

    '''
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print (model.components_)


#--------------------------------------------------------------------------------------
    
def ldaAnalysis(docLst,filename_output_txt='',n_topics=45,n_top_words=50,n_iteration=100,n_features=2000):
    '''
    Make LDA analysis on all processed news.
    Args:
    docLst(list): a list of processed news in same json file.
    docLst is like ['processed news1','processed news2',......]
    '''
    t0 = time()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(docLst)

    lda = LatentDirichletAllocation(n_components=n_topics, 
                                max_iter=n_iteration,
                                learning_method='batch')
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    perplexity = lda.perplexity(tf)
    actual_iteration = lda.n_iter_
   
    if filename_output_txt:
        
        top_words_list = getTopWordsList(lda,tf_feature_names, n_top_words)
        
        with open(filename_output_txt,'w') as f:
    
            f.write('Max Iteration:  '+str(n_iteration)+'\n')
            f.write('Actual Iteration:  '+str(actual_iteration)+'\n')
            f.write('Number of Topics:  '+str(n_topics)+'\n')
            f.write('Number of Top Words:  '+str(n_top_words)+'\n')
            f.write('Perplexity:  '+str(perplexity)+'\n')
            
            for i in range(len(top_words_list)):
                top_words = top_words_list[i]
                
                f.write('Topic '+str(i+1)+'\n')
                f.write(top_words+'\n')
                
    else:
        print('Perplexity:  '+str(perplexity)+'\n')
        print_top_words(lda,tf_feature_names, n_top_words)
        print(time()-t0)
        
    return perplexity,actual_iteration
    print('Done*********************************************************')
    
    

#-----------------------------------------------------------------------------
            
def ldaOptimization(docLst,
                    filepath_output_txt,
                    filename_output_csv,
                    n_topics_list,
                    fixed_top_words=100,
                    fixed_iteration = 500,
                    fixed_features=2000):
    
    output=pd.DataFrame()
    output.to_csv(filename_output_csv)
    
    i = 0
    perplexity_list = []
    for n in n_topics_list:
        print('************NEXT*****************')
        print(n)
        
        t0 = time()
        i+=1
        
        filename = filepath_output_txt+str(n)+'_topics.txt'
        
        perplexity,actual_iteration = ldaAnalysis(docLst,
                                                  filename_output_txt = filename,
                                                  n_topics = n,
                                                  n_top_words = fixed_top_words,
                                                  n_iteration = fixed_iteration,
                                                  n_features = fixed_features)
        print('  txt saved')
        print(str(time()-t0)+'  Seconds')
        perplexity_list.append(perplexity)

        output = pd.read_csv(filename_output_csv,index_col=0)
        output.loc[str(i),'Number of Topics'] = n
        output.loc[str(i),'Number of Top Words'] = fixed_top_words
        output.loc[str(i),'Perplexity'] = perplexity
        output.loc[str(i),'Max Iteration'] = fixed_iteration
        output.loc[str(i),'Actual Iteration'] = actual_iteration
        output.loc[str(i),'Seconds Needed'] = time()-t0
        output.to_csv(filename_output_csv)
        
        print('  csv saved --')
    
    best_n_topics = n_topics_list[perplexity_list.index(min(perplexity_list))]

    filename = filepath_output_txt+'optimization.txt'
    ldaAnalysis(docLst,
            filename_output_txt = filename,
            n_topics = best_n_topics,
            n_top_words = fixed_top_words,
            n_iteration = fixed_iteration,
            n_features = fixed_features)


    print('ALL DONE')

    
#####################################################################

def main():
    pickle_1 = '/data/output_news/preprocess_10car.pickle'
    pickle_2 = '/data/output_news/preprocess_20car.pickle'

    filepath_output_txt_1 = '/data/output_news/topics_10car/'
    filepath_output_txt_2 = '/data/output_news/topics_20car/'

    filename_output_csv_1 = '/data/output_news/optimization_10car.csv'
    filename_output_csv_2 = '/data/output_news/optimization_20car.csv'

    docLst_1= combineNewsViaPickle(pickle_1)
    docLst_2= combineNewsViaPickle(pickle_2)


    n_topics_list = list(range(30,85,5))

    ldaOptimization(docLst_1,
                    filepath_output_txt_1,
                    filename_output_csv_1,
                    n_topics_list,
                    fixed_top_words=100,
                    fixed_iteration =300,
                    fixed_features=2000)

    ldaOptimization(docLst_2,
                    filepath_output_txt_2,
                    filename_output_csv_2,
                    n_topics_list,
                    fixed_top_words=100,
                    fixed_iteration =300,
                    fixed_features=2000)

if __name__ == '__main__':
    main()

