import numpy as np
import pandas as pd
import pickle
def get_countvec(data):
    #make a count vectorizer with the extracted words.

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='word')
    count_fit = vectorizer.fit(data)
    # tokenize and build vocab
    #vectorizer.fit(text)
    # summarize
    print(len(vectorizer.vocabulary_))
    print(vectorizer.vocabulary_)
    # encode document
    #vector = vectorizer.transform(text)
    return vectorizer,count_fit


def get_tfidvec(data,ngram=True,max_features=400):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if ngram:
        #use this as an example for transforming the inputs to model
        #xtrain_tfidf =  tfidf_vect.transform(X_train)
        #xvalid_tfidf =  tfidf_vect.transform(X_test)
        # ngram level tf-idf 
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=max_features)
        tfid_fit_ngram = tfidf_vect_ngram.fit(data)

        #print(len(tfidf_vect_ngram.vocabulary_))
        #print(tfidf_vect_ngram.vocabulary_)
        #use this as an example for transforming the inputs to model
        #xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
        #xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
        return tfidf_vect_ngram, tfid_fit_ngram
    
    
    # In tis block and next we apply TFIDF for :
    # a. Word Level TF-IDF : Matrix representing tf-idf scores of every term in different documents
    # b. N-gram Level TF-IDF : N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams


    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=max_features)
    tfid_fit = tfidf_vect.fit(data)

    tfidf_dict  = tfidf_vect.vocabulary_

    #print(len(tfidf_dict))
    #print(tfidf_dict)
    return tfidf_vect, tfid_fit

def get_countvec_dataframe(data,cpgrams_df,col_name='subject_content'):
    print("Running Count Vectorizer on:",col_name)
    vectorizer, count_fit = get_countvec(data)
    
    #applying the transformation to all three vectors 
    countVec = vectorizer.transform(cpgrams_df[col_name].apply(lambda x: " " if x is np.NaN else x))
    #Creating three data frames
    count_df = pd.DataFrame(countVec.toarray(), columns=vectorizer.get_feature_names())
    return count_df,vectorizer,count_fit 

def get_tfidvec_dataframe(data,cpgrams_df,col_name='subject_content',max_features=400):
    if 'the' in data:
        print("Stopword in the Vectorizer")
        return
    print("Running tfid Vectorizer on:",col_name,max_features)
    tfidf_vect, tfid_fit = get_tfidvec(data,ngram=False)
    #applying the transformation to all three vectors 
    tfidfVec = tfidf_vect.transform(cpgrams_df[col_name].apply(lambda y: " " if y is np.NaN else y))
    #Creating three data frames
    tfidf_df = pd.DataFrame(tfidfVec.toarray(), columns=tfidf_vect.get_feature_names())
    return tfidf_df,tfidf_vect, tfid_fit

def get_tfid_ngram_vec_dataframe(data,cpgrams_df,col_name='subject_content',max_features=400):
    print("Running tfid ngram Vectorizer on:",col_name,max_features)
    tfidf_vect_ngram, tfid_fit_ngram = get_tfidvec(data,ngram=True)
    #applying the transformation to all three vectors 
    ngramVec = tfidf_vect_ngram.transform(cpgrams_df['subject_content'].apply(lambda z: " " if z is np.NaN else z))
    #Creating three data frames
    ngram_df = pd.DataFrame(ngramVec.toarray(), columns=tfidf_vect_ngram.get_feature_names())
    return ngram_df,tfidf_vect_ngram, tfid_fit_ngram

def get_df_for_wordtovec(data,df,technique='count',col_name='subject_content',max_features=400): #'tfid','ngram'
    if technique=='tfid':
        tfidf_df,tfidf_vect, tfid_fit = get_tfidvec_dataframe(data,df,col_name,max_features)
        # Let's save our vectorizers locally
        with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "wb") as f:
            pickle.dump(tfidf_vect, f)
        return tfidf_df
    elif  technique=='ngram':
        ngram_df,tfidf_vect_ngram, tfid_fit_ngram = get_tfid_ngram_vec_dataframe(data,df,col_name,max_features)
        with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "wb") as f:
            pickle.dump(tfidf_vect_ngram, f)
        return ngram_df
    else:
        count_df,vectorizer,count_fit  = get_countvec_dataframe(data,df,col_name)
        with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "wb") as f:
            pickle.dump(vectorizer, f)
        return count_df

    
def get_wordtovec(data,df,technique='count',col_name='subject_content',max_features=400): #'tfid','ngram'
    if technique=='tfid':
        tfidf_df,tfidf_vect, tfid_fit = get_tfidvec_dataframe(data,df,col_name,max_features)
        # Let's save our vectorizers locally
        with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "wb") as f:
            pickle.dump(tfidf_vect, f)
        return tfidf_df,tfidf_vect, tfid_fit
    elif  technique=='ngram':
        ngram_df,tfidf_vect_ngram, tfid_fit_ngram = get_tfid_ngram_vec_dataframe(data,df,col_name,max_features)
        with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "wb") as f:
            pickle.dump(tfidf_vect_ngram, f)
        return ngram_df,tfidf_vect_ngram, tfid_fit_ngram
    else:
        count_df,vectorizer,count_fit  = get_countvec_dataframe(data,df,col_name)
        with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "wb") as f:
            pickle.dump(vectorizer, f)
        return count_df,vectorizer,count_fit 

