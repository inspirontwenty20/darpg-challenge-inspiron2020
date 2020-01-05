#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports and downloads that NLTK needs
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import string
from nltk.corpus import words as words_lib
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
#remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#adding few downloads for lematizer to work

#nltk.download('punkt', download_dir='/tmp/')  #  Tokenizing text strings
#nltk.download('wordnet', download_dir='/tmp/')  # Large lexical database of the English language
#nltk.download('stopwords', download_dir='/tmp/')  # Dictionary of stopwords
#nltk.data.path.append("tmp")


# In[ ]:


#Various functions for text preprocessing

from nltk.probability import FreqDist
import nltk
def get_most_common_features(wordfreq,word_threhold = 20):
    fdist1 = FreqDist(wordfreq)
    #print(fdist1)
    return fdist1.most_common(word_threhold)
def get_bagofwords_freq(corpus):
    #remove stop words like a, this, etc.
    wordfreq = []
    for w in corpus: 
        # a lot of redacted text contained . and *
        if w not in stop_words:#,'[\.]*','[\*]*']:
            wordfreq.append(w)
        #wordfreq.append(w)
    #print(wordfreq)
    
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer()
    X = vec.fit_transform(wordfreq)
    #requests_df.loc[:,'subject_content'])
    #print(vec.get_feature_names())
    #print(X.toarray())
   
    
    most_freq_words = get_most_common_features(wordfreq)
    return vec.get_feature_names(),most_freq_words,X.toarray()


def tfid(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    transformer = TfidfVectorizer(smooth_idf=False)
    tidf_X = transformer.fit_transform(corpus)
    tfid_df =  pd.DataFrame(tidf_X.toarray(), columns=transformer.get_feature_names())
    #return tidf_X.toarray()
    return tfid_df


def number_remover(string):
    import re
    #removing all numbers
    regex = re.compile('[0-9]+')
    #regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')  # Regex to remove HTML tags and entities
    text_without_num = re.sub(regex, ' ', string)  # Replace HTML tags and entities with single space
    return text_without_num # Return text cleaned of HTML tags and entities



# In[ ]:


#copying the code from http://norvig.com/spell-correct.html for spelling correction. 
# will ltry this. We can add the department abbrevation (Code) to the training set (Big.txt with some weight)
# to make sure we are getting right department too. 


import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('./../Dictionaries/big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def stem_porter_text(text):
    from nltk.stem import PorterStemmer # for Lancaster algorthm, use "from nltk.stem import LancasterStemmer"
    porter = PorterStemmer()
    return([porter.stem(w) for w in text])


def stem_word(word):    
    from nltk.stem import PorterStemmer as ps
    return ps().stem(word)

# Lemmatize
## Create lemmatize function for verbs ('v')

def lemmatize_text(text):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return([lemmatizer.lemmatize(w, pos = 'v') for w in text]) # lemmatize verbs (pos = 'v')

def lemmatize_word(word):
    from nltk.stem import WordNetLemmatizer
    #lemmatizer = nltk.stem.WordNetLemmatizer()
    checkList=[WordNetLemmatizer().lemmatize(word,'v'),
               WordNetLemmatizer().lemmatize(word,'n'),
               WordNetLemmatizer().lemmatize(word,'r'),
               WordNetLemmatizer().lemmatize(word,'a')]
    if most_common(checkList) == word:
        return stem_word(word)
    return most_common(checkList)

def most_common(lst):
    #print(lst)
    return max(set(lst), key=lst.count)
    

def remove_special_characters(string):
    # a lot of redacted text contained . and *
    #string = string.replace("[^A-Za-z0-9]","")
    import re
    regex = re.sub('[^A-Za-z0-9]+', '', string)
    string = re.sub(regex, ' ', string)
    return string

def remove_stop_words(string):
    #from nltk.corpus import stopwords
    #stop_words = set(stopwords.words('english'))
    string_without_stopwords = []
    for word in string.split(): 
        if word not in stop_words:
            string_without_stopwords.append(word)
    return " ".join(string_without_stopwords)


word_dict = set(words_lib.words())
punct = str.maketrans(dict.fromkeys(string.punctuation))

def pre_process_text(df,col_name='subject_content'):
     
    all_data = []
    special_words = []
    counter = 0
    for string_data in df[col_name]:
        counter +=1
        #if counter >10:
        #    break
        #print(string_data)
        if string_data is np.NaN:
            print("Nan so continue",df.loc[counter:counter,'encoded_y'])
            converted_string.append('None')
            all_data.append(" ".join(converted_string))
            continue
        string_data = string_data.lower()
        string_data = remove_special_characters(string_data)
        string_data = string_data.translate(punct)
        string_data = number_remover(string_data)
        
        string_data = remove_stop_words(string_data)
        #print(string_data)
        #print(".................>>>>>>>")
        #if 'and' in string_data:
        #    print('!Stopword not removed')
        converted_string = []
        for word in string_data.split():
            # first lemmatize : since the other words are being ignored by the words.words()
            word = lemmatize_word(word)
            #print(word)
            #spell check - correction
            #word = nltk_helper.correction(word)
            if word not in word_dict:
                special_words.append(word)
                continue
            if len(word)>2:
                #Ensuring that the word we are adding are not random characters as seen in the text.
                #synonym = wn.synsets(word)
                #print(word)
                #print(synonym)
                #if word in synonym:
                    #writeSynonym = filename.replace(str(word), str(synonym[0]))
                converted_string.append(word) 
        print(".",end=" ")
        converted_string = list(set(converted_string))
        all_data.append(" ".join(converted_string))
    return all_data,special_words


