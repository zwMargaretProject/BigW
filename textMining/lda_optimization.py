import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time
import pickle

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
    
def ldaAnalysis(docLst,
                n_topics,
                n_top_words,
                n_iteration,
                n_features):
    '''
    Make LDA analysis on all processed news.
    Args:
    docLst(list): a list of processed news in same json file.
    docLst is like ['processed news1','processed news2',......]
    '''
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(docLst)

    lda = LatentDirichletAllocation(n_components=n_topics, 
                                    max_iter=n_iteration,
                                    learning_method='batch')
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    top_words_list = getTopWordsList(lda,tf_feature_names, n_top_words)
    perplexity = lda.perplexity(tf)
    
    
    return lda,top_words_list,perplexity,tf_vectorizer.max_features

    

#-----------------------------------------------------------------------------
            
def ldaOptimization(docLst,
                    filename_output_txt,
                    filename_output_csv,
                    n_topics_list,
                    fixed_top_words,
                    fixed_iteration ,
                    fixed_features,
                    pickle_all_records='',
                    pickle_all_lda=''):
    
    output=pd.DataFrame()
    output.to_csv(filename_output_csv)
    
    perplexity_list = []
    all_top_words_list = []
    
    i=0
    lda_all_list = []
    
    for n in n_topics_list:
        print('max features: '+str(fixed_features))
        i+=1
        print(str(i)+' Try:  '+str(n)+' Topics')
        t0 = time()
        
        lda,top_words_list,perplexity,model_features = ldaAnalysis(docLst,
                                                       n_topics = n,
                                                       n_top_words = fixed_top_words,
                                                       n_iteration = fixed_iteration,
                                                       n_features = fixed_features)

        
        lda_all_list.append(lda)

        perplexity_list.append(perplexity)
        all_top_words_list.append(top_words_list)

        output = pd.read_csv(filename_output_csv,index_col=0)
        output.loc[str(i),'Number of Topics'] = n
        output.loc[str(i),'Number of Top Words'] = fixed_top_words
        output.loc[str(i),'Perplexity'] = perplexity
        output.loc[str(i),'Max Iteration'] = fixed_iteration
        output.loc[str(i),'Actual Iteration'] = lda.n_iter_
        output.loc[str(i),'Seconds Needed'] = time()-t0
        output.to_csv(filename_output_csv)
        
        print('  csv saved --')
    
    idx = perplexity_list.index(min(perplexity_list))
    best_n_topics = n_topics_list[idx]
    best_top_words_list = all_top_words_list[idx]
    perplexity = perplexity_list[idx]

    if pickle_all_records:
        f = open(pickle_all_records,'wb')
        pickle.dump(perplexity_list,f)
        pickle.dump(all_top_words_list,f)
        f.close()
        print('  Pickle Saved')
    
    
    with open(filename_output_txt,'w') as f:
    
        f.write('Max Iteration:  '+str(fixed_iteration)+'\n')
        f.write('Actual Iteration:  '+str(lda.n_iter_)+'\n')
        f.write('Number of Topics:  '+str(best_n_topics)+'\n')
        f.write('Number of Top Words:  '+str(fixed_top_words)+'\n')
        f.write('Perplexity:  '+str(perplexity)+'\n')

        
        for i in range(len(best_top_words_list)):
            top_words = best_top_words_list[i]
            f.write('\n')
            f.write('Topic '+str(i+1)+'\n')
            f.write(top_words+'\n')
            
        print('  Optimization Saved ')

    if pickle_all_lda:
        f = open(pickle_all_lda,'wb')
        for i in range(len(lda_all_list)):
            lda_class = lda_all_list[i]
            pickle.dump(lda_class,f)
        f.close()

        print('  LDA Pickle Saved')
    
            
    
    print('ALL DONE--------------')

    return best_n_topics,model_features
