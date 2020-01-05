#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.model_selection import train_test_split
import pandas as pd
import helper_import_path
import model_builder_helper as mbh


all_original_data = pd.read_csv("./../Data/merged_eaol_reduced_classes_30.csv")
y_colname = 'Apex_org_name'#'Parent_org_name' #

technique='tfid' #count,ngram
# In[10]:


all_regs_for0 = all_original_data.loc[all_original_data['encoded_y'].apply(lambda x:x==0),'registration_no']

import random
random.seed(4)
#base_reg_list = set(all_original_data['registration_no']) # why EAOL since those are the tickets we are targetting.
regids_to_exclude = random.sample(set(all_regs_for0), 7500)

all_original_data = all_original_data.loc[all_original_data['registration_no'].apply(lambda x:x not in regids_to_exclude),:]


# In[11]:


#all_original_data.shape
all_original_data.reset_index(inplace=True,drop=True)


# In[12]:


all_original_data


# In[13]:


all_original_data.columns
selected_columns = ['registration_no','encoded_y', 'country_name','state_name', 'subject_content',y_colname]


    # In[14]:


import random
random.seed(4)
base_reg_list = set(all_original_data['registration_no']) # why EAOL since those are the tickets we are targetting.
test_reg_list = random.sample(base_reg_list, 3000)
train_reg_list = list(set(all_original_data['registration_no'])-set(test_reg_list))
len(train_reg_list)




def model_data_prep():

        # #### Select and use train data

    # In[15]:


    #Selected Columns 
    train_reg_data = all_original_data.loc[all_original_data['registration_no'].apply(lambda x: x in train_reg_list),selected_columns]
    train_reg_data.shape


    # In[16]:



    print("Train Data, number of rows before drop na",train_reg_data.shape)
    na_free = train_reg_data.dropna()
    only_na = train_reg_data[~train_reg_data.index.isin(na_free.index)]
    print("Train Data, number of rows After drop na",na_free.shape)


    # In[17]:


    only_na


    # In[18]:


    train_reg_data.columns
    train_reg_data.reset_index(inplace=True,drop=True)
    train_reg_data.loc[0:0,:]


    # In[19]:


    print(train_reg_data.shape)


    # In[20]:


    #print(len(data))


    # In[21]:


    import nltk_helper_functions as nltk_helper
    import word2vec_helper as w2v 
    import pickle

    # So now we have CountVectorizer, TF-IDF word, and TF-IDF n-grams level.
    # use each of them tro preare 3 data sets, and use for training. 
    # Assess what gives the best results. 
    ## Preprocess the subject content
    data , special_words = nltk_helper.pre_process_text(train_reg_data)
    technique='tfid' #,ngram, , count
    max_features = 1000
    print(len(data))
    train_reg_data['subject_content'] = data
    wordtovec_df, vectorizer, count_fit = w2v.get_wordtovec(data,train_reg_data,technique,col_name='subject_content',max_features=max_features)

    #vectorizer.transform()


    # In[22]:


    print(wordtovec_df.shape)
    data_matrix = count_fit.fit_transform(data)


    # In[23]:


    data_dense = data_matrix.todense()
    #TODO: Confirm if higher is better -> looks like it.
    #Compute Sparsicity = Percentage of Non-Zero cells
    print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")


    # In[24]:


    train_reg_data.shape


    # In[25]:

    encoder_objects = []
    col_names = ['country_name','state_name'] #'distname', 
    # removed this since many of these values were NaN for test data and we had a limited set. 
    #TODO*Check feature importance to select/deselect this in future.
    for col_name in col_names:
        train_reg_data[col_name],label_encoder_y = mbh.get_encoded_values(train_reg_data[col_name].fillna('X'))
        encoder_objects.append(label_encoder_y)
        # Persiting the encoder so, we can use it later to decode.
        with open("./../TrainedVectors/labelencoder_"+col_name+".pickle", "wb") as f:
            pickle.dump(label_encoder_y, f)


    # In[58]:


    ## Test labelencoder

    #with open("./../TrainedVectors/labelencoder_country_name.pickle", "rb") as f:
    #    encoder = pickle.load(f)
        #assert encoder.transform(['India']) in [9], " Encoder not working properly or training set is different."
        #assert encoder.inverse_transform([9]) in ['India'],"Inverse transform is not correct."
    #encoder.transform(['India'])


    # In[59]:


    #with open("./../TrainedVectors/labelencoder_state_name.pickle", "rb") as f:
    #    encoder = pickle.load(f)
        #assert encoder.transform(['India']) in [9], " Encoder not working properly or training set is different."
        #assert encoder.inverse_transform([9]) in ['India'],"Inverse transform is not correct."
    #encoder.transform(['Karnataka'])


    # In[28]:


    wordtovec_df[train_reg_data.columns] = train_reg_data.loc[:,:]


    # In[29]:


    #train_reg_data['country_name']


    # In[30]:


    print(wordtovec_df.shape)

    #wordtovec_df.dropna(inplace=True)
    #wordtovec_df.head()
    print(wordtovec_df.columns)


    # In[31]:


    #train_reg_data.dropna(inplace=True)
    ## Selecting the x_columns
    #print(train_reg_data.shape)
    #print(train_reg_data.columns)
    colnames = wordtovec_df.columns
    y_column = 'encoded_y' #right now this controlled in the processing

    x_y_cols = list(set(colnames) - set([y_colname,'registration_no','subject_content','distname']))
    x_cols = list(set(x_y_cols) - set([y_column]))
    #x_cols


    # In[32]:


    #x_cols


    # In[33]:


    # check causes of null
    #wordtovec_df['encoded_y'][wordtovec_df['encoded_y'].apply(lambda x: x is np.NaN)]
    #wordtovec_df['country_name'][wordtovec_df['country_name'].apply(lambda x: x is np.NaN)]
    #dist_nan = wordtovec_df['distname'][wordtovec_df['distname'].apply(lambda x: x is np.NaN)]  # this one has NaN
    #print(dist_nan.shape) #1313
    #wordtovec_df['state_name'][wordtovec_df['state_name'].apply(lambda x: x is np.NaN)]
    #subject_nan = wordtovec_df['subject_content'][wordtovec_df['subject_content'].apply(lambda x: x is np.NaN)]
    #print(subject_nan.shape) #911


    # In[34]:


    train_reg_data = wordtovec_df.loc[:,x_y_cols] #copy and not ref
    print("Train Data, number of rows before drop na",train_reg_data.shape)
    na_free = train_reg_data.dropna()
    only_na = train_reg_data[~train_reg_data.index.isin(na_free.index)]
    print("Train Data, number of rows After drop na",na_free.shape)


    # In[35]:


    na_free.shape
    #train_reg_data.reset_index(inplace=True,drop=True)
    #train_reg_data.loc[0:0,:]


    # In[36]:


    train_reg_data = na_free.loc[:,:]
    #train_reg_data


    # In[37]:


    #writing the file to CSV to save all the preprocessing steps for model training. 
    #train_reg_data.to_csv("./../Data/merged_eaol_processed_train_data.csv",index=False)


    # In[38]:


    # commenting out the old dataframe and using the new one without int. 
    X = train_reg_data[x_cols]
    y = train_reg_data[y_column]
    #X = X.loc[0:1000, :]
    #y = y.loc[0:1000]
    # Train test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
    return X_train, X_test, y_train, y_test, encoder_objects

# #### Model Selection

# A model with higher log-likelihood and lower perplexity (exp(-1. * log-likelihood per word)) is considered to be good. Let’s check for our model.

# In[39]:


def get_lda_model(X,y):
    
    from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                          max_iter=10,               # Max learning iterations
                                          learning_method='online',   
                                          random_state=100,          # Random state
                                          batch_size=128,            # n docs in each learning iter
                                          evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               # Use all available CPUs
                                         )
    lda_output = lda_model.fit_transform(X,y)

    print(lda_model)  # Model attributes
    from pprint import pprint
    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(X,y))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    #this is giving some error
    #print("Perplexity: ", lda_model.perplexity(X,y))

    # See model parameters
    pprint(lda_model.get_params())
    return lda_model


# The most important tuning parameter for LDA models is n_components (number of topics). In addition, I am going to search learning_decay (which controls the learning rate) as well.
# 
# Besides these, other possible search params could be learning_offset (downweigh early iterations. Should be > 1) and max_iter. These could be worth experimenting if you have enough computing resources.
# 
# Be warned, the grid search constructs multiple LDA models for all possible combinations of param values in the param_grid dict. So, this process can consume a lot of time and resources.

# In[ ]:


def run_GSCV_lda(X,y):
    from sklearn.model_selection import GridSearchCV
    from pprint import pprint
    # Define Search Param
    search_params = {'n_components': [20, 25, 30], 'learning_decay': [.5, .7, .9]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    return model.fit(X,y)


# In[ ]:


def run_decision_tree_model(X_train,y_train):
    print("Decision Tree Model")
    model = mbh.get_decision_tree_model(X_train,y_train)
    pred_train = mbh.get_prediction(model,X_train)
    pred_test = mbh.get_prediction(model,X_test)
    #Train Metrics
    mbh.get_metrics(pred_train,y_train,"Train")
    #Test Metrics
    mbh.get_metrics(pred_test,y_test,"Test")
    return model


# In[ ]:


def run_random_forest_model(X_train,y_train):
    print("------------------------------------------------")
    print("Random Forest Model")
    model = mbh.get_randomforest_model(X_train,y_train)
    pred_train = mbh.get_prediction(model,X_train)
    pred_test = mbh.get_prediction(model,X_test)
    #Train Metrics
    mbh.get_metrics(pred_train,y_train,"Train")
    #Test Metrics
    mbh.get_metrics(pred_test,y_test,"Test")
    return model, y_test, pred_test


# In[40]:


#adding code here for NB
def run_NaiveBayes_GS(X_train,y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.naive_bayes import MultinomialNB  
    
    clf = MultinomialNB()
    # using X_train_dtm (timing it with an IPython "magic command")
    
    #get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')
    clf = clf.fit(X_train, y_train)
    y_test, y_pred = None, None
    #y_pred = clf.predict(X_test)
    #from sklearn import metrics
    #print(metrics.accuracy_score(y_test, y_pred))
    return clf,y_test, y_pred
    


# In[ ]:


def run_GSCV_different_models(X_train,y_train):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier


    # Set the parameters by cross-validation for RF
    #print("using RF")
    #tuned_parameters = { 
    #    'n_estimators': [max_features],
    #    'max_features': ['auto', 'sqrt', 'log2'],
    #    'max_depth' : [8, 12],
    #    'criterion' :['gini', 'entropy']
    #}
    
    # Set the parameters by cross-validation for SVC
    print("using SVC")
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                 'C': [100, 1000]},
    #                {'kernel': ['linear'], 'C': [100, 1000]}]
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4],
                     'C': [ 1000]}]
    #                {'kernel': ['linear'], 'C': [100, 1000]}]
    scores =  ['recall'] #['precision']#,
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        print('.',end='')
        clf = GridSearchCV(
            SVC(probability=True), tuned_parameters, scoring='%s_macro' % score, cv=2
        )
        #clf = GridSearchCV(
        #    RandomForestClassifier(random_state=42,class_weight="balanced"), tuned_parameters, scoring='%s_macro' % score, cv=5
        #)
        clf_fit = clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        
        return clf,clf_fit,y_true, y_pred


# In[ ]:





# In[41]:


#list(set(X_train.columns) - set(vectorizer.get_feature_names()))


# In[42]:


#model, y_true, y_pred = run_random_forest_model(X_train,y_train)


# In[43]:





def driver_code_get_model():
    X_train, X_test, y_train, y_test, encoder_array = model_data_prep()
    model,y_true,y_pred = run_NaiveBayes_GS(X_train,y_train)
    print("running prediction on test data")
    predict_on_test_data(model,encoder_array,selected_columns,all_original_data,test_reg_list)
    return model,encoder_array


# In[ ]:


#model,clf_fit, y_true,y_pred = run_GSCV_different_models(X_train,y_train)


# In[ ]:


#print(get_confusion_metric(y_test, pred_test))

#below functions dont work good for multiclass problem, hence n0ot to be used.  
#print(get_auc_roc(model, X_test,y_test))
#get_false_true_positive_rates(model, X_test, y_test)


# In[ ]:


#cm = mbh.get_PyCM(y_test, pred_test)

##
### In[44]:
##
##
##import pickle 
##filename = 'model_NB_111.pickle'
##pickle.dump(model, open(filename, 'wb'))
###some time later...
## 
###load the model from disk
##loaded_model = pickle.load(open(filename, 'rb'))
##result = loaded_model.score(X_test, y_test)
##print(result)
##
##
### In[55]:
##
##
##len(x_cols)
##text
##
##
### In[62]:
##
##
#### testing code
##import pickle
##filename = './model_NB_111.pickle'
##loaded_model = pickle.load(open(filename, 'rb'))
##
##with open("./../TrainedVectors/"+'tfid'+"_vectorizer.pickle", "rb") as f:
##    vectorizer = pickle.load(f) 
##
##test_string =text
###"Subject delay to delivery for online booking Booking date ** ** **** CUNJUMER NO IS **********SIR  My name is Roji khatoon and I am customer of Uchkagaon Indian amtha bhauan line bazar but I am very upset with your service at I had online booking this ** ** **** but not arrive my LPG cylinder booking order no is RE************** my reference number is ********* but my cylinder is not arrive at time      I requested to you my cylinder delivery on time Thank you NAME ROJI KHATOONConsumer no is ********** Branch UCHKAGAON INDANE AMTHA LINE BAZAR GOPALGANJ BIHAR REG MO NU IS **********"
##temp_vec = vectorizer.transform([test_string])# .apply(lambda x: " " if x is np.NaN else x))
##temp_df = pd.DataFrame(temp_vec.toarray(), columns=vectorizer.get_feature_names())  
##temp_df['country_name'] = 7
##temp_df['state_name'] = 16
###X_test = pd.read_csv("./../TrainedVectors/model_NB_data.csv")
##y_test = loaded_model.predict(temp_df) #temp_df) #
##print(y_test)
###(temp_df-pred_wordtovec_df.loc[0:0,x_cols]).sum(axis = 1)
##
##
### In[64]:
##
##
##
##
##
### In[ ]:
##
##
###feat_imp = pd.DataFrame(
###            {
###                'feature': list(X_test.columns.values),
###                'importance': clf_fit.feature_importances_
###            }
###        )
###finding out important featuresand try to eliminate few later. 
##
###feat_imp.sort_values('importance', ascending=False).head(20)
##
##
def predict_on_test_data(model,encoder_array,selected_columns,all_original_data,test_reg_list):
    # ### Predict on test data

    # In[56]:


    test_reg_data = all_original_data.loc[all_original_data['registration_no'].apply(lambda x: x in test_reg_list),selected_columns]
    test_reg_data.shape
    test_reg_data.reset_index(inplace=True,drop=True)
    test_reg_data.loc[0:15,:]

    #Write test data to csv for reference
    test_reg_data.to_csv('./../reports/test_data_for_prediction.csv',index=False)

    # In[57]:


    #text = test_reg_data.loc[3,'subject_content']


    # In[43]:


    #Preprocess data
    import pickle
    import nltk
    import helper_import_path 
    import nltk_helper_functions as nltk_helper

    import word2vec_helper as w2v 
    import numpy as np

    with open("./../TrainedVectors/"+technique+"_vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f) 


    # In[44]:


    ## Conclusion : removing distname as it is removing multiple roles ? Check if this was an important feature later.
    #       'distname', 'state_name', 'subject_content'
    #test_reg_data['encoded_y'][test_reg_data['encoded_y'].apply(lambda x: x is np.NaN)]
    #test_reg_data['country_name'][test_reg_data['country_name'].apply(lambda x: x is np.NaN)]
    #dist_nan = test_reg_data['distname'][test_reg_data['distname'].apply(lambda x: x is np.NaN)]  
    #print(dist_nan.shape)# this one has NaN 40
    #test_reg_data['state_name'][test_reg_data['state_name'].apply(lambda x: x is np.NaN)]
    #test_reg_data['subject_content'][test_reg_data['subject_content'].apply(lambda x: x is np.NaN)]


    # In[45]:



    #vectorizer.transform()
    #Prediction code
    vec = vectorizer.transform(test_reg_data['subject_content'] .apply(lambda x: " " if x is np.NaN else x))
    pred_wordtovec_df = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names())  


    # In[46]:


    #test_reg_data['country_name']
    pred_wordtovec_df.shape


    # In[47]:


    # Handling nulls
    null_columns=pred_wordtovec_df.apply(lambda x: 0 if x is np.NaN else x)
    null_columns.shape
    null_columns.head()


    # In[48]:


    #Replace nan by 0 Not sure if this is correct.
    #print("Shape before replace NaN",pred_wordtovec_df.shape)
    #wordtovec_df = wordtovec_df.replace(np.NaN, 0)
    #print("Shape After replace NaN",pred_wordtovec_df.shape)


    # In[49]:


    # Handling nulls

    #inf_columns=np.isfinite(pred_wordtovec_df).all()
    #print("Shape before replace -inf",pred_wordtovec_df.shape)
    #wordtovec_df = wordtovec_df.replace(-np.Inf, 0)
    #print("Shape after replace inf",pred_wordtovec_df.shape)
    #pred_wordtovec_df.replace(np.Inf, 0)
    #inf_columns=np.isfinite(pred_wordtovec_df).all()


    # In[50]:


    #np.isfinite(pred_wordtovec_df).all()


    # In[51]:


    import model_builder_helper as mbh

    # Encoding values for the following 
    col_names = ['country_name', 'state_name'] #'distname',
    for col_name_index in range(len(col_names)):
        # Read Persited encoder so, we can use it later to decode.
        encoder = encoder_array[col_name_index]
        test_reg_data[col_names[col_name_index]] = test_reg_data[col_names[col_name_index]].map(lambda s: '<unknown>' if s not in encoder.classes_ else s)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')
        test_reg_data[col_names[col_name_index]] = encoder.transform(test_reg_data[col_names[col_name_index]])
##        with open("./../TrainedVectors/labelencoder_"+col_name+".pickle", "rb") as f:
##            encoder = pickle.load(f)
##            #test_reg_data[col_name] = encoder.transform(test_reg_data[col_name])
##            test_reg_data[col_name] = test_reg_data[col_name].map(lambda s: '<unknown>' if s not in encoder.classes_ else s)
##            encoder.classes_ = np.append(encoder.classes_, '<unknown>')
##            test_reg_data[col_name] = encoder.transform(test_reg_data[col_name])


    # In[52]:


    #test_reg_data[col_name]
    columns_selected_eaol_data = list(set(test_reg_data.columns)-set(['registration_no', y_colname,'distname', 'subject_content']))
    pred_wordtovec_df[columns_selected_eaol_data] = test_reg_data.loc[:,columns_selected_eaol_data]


    # In[53]:


    pred_wordtovec_df.shape


    # In[54]:


    #print(model.n_features_)
    y_column = 'encoded_y' #right now this controlled in the processing

    x_y_cols = list(set(pred_wordtovec_df.columns) - set([y_colname,'registration_no','subject_content','distname']))
    x_cols = list(set(x_y_cols) - set([y_column]))
    #print(len(x_cols_pred))
    #print(len(x_cols))
    #print("Shape of input vec ", pred_wordtovec_df.shape)


    # In[55]:


    #print('registration_no' in x_cols_pred )
    #print(set(x_cols_pred)-set(x_cols))
    #print(x_cols)


    # In[56]:


    #print("Number of rows before dropping na",pred_wordtovec_df.shape)
    #wordtovec_df.dropna(inplace=True)
    na_free = pred_wordtovec_df.dropna()
    only_na = pred_wordtovec_df[~pred_wordtovec_df.index.isin(na_free.index)]
    #wordtovec_df.describe()
    #print("Number of rows After dropping na",na_free.shape)
    #wordtovec_df[x_cols].head()


    # In[57]:


    pred_wordtovec_df = na_free.loc[:,:]
    #print("Final pred DF shape",pred_wordtovec_df.shape)


    # In[83]:


    pred_wordtovec_df.head(15)


    # In[69]:



    def plot_freq(merged_df,col_name='encoded_y'):
        import matplotlib.pyplot as plt
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.hist(merged_df[col_name])#,bins=len(merged_df['encoded_y'].value_counts()))
        plt.show()
    #plot_freq(pred_wordtovec_df)


    # In[ ]:


    #load the model from disk to do the prediction and scoring
    #filename = 'model_RF_5.pickle'
    #model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    #print(result)


    # In[58]:


    pred = mbh.get_prediction(model,pred_wordtovec_df[x_cols])
#    print("Predicted values",pred)
    #print("Number of predicted values",len(pred))
    pred = pred.astype(int)


    # In[59]:


    #print(pred)


    # In[60]:


    with open("./../TrainedVectors/labelencoder_y.pickle", "rb") as f:
        encoder = pickle.load(f)
    pred_labels = encoder.inverse_transform(pred)


    # In[61]:



    actual_labels = encoder.inverse_transform(pred_wordtovec_df['encoded_y'].astype(int))


    # In[62]:


    outcome = []
    for index in range(0,len(pred_labels)):
        if actual_labels[index] in pred_labels[index]:
            outcome.append(1)
        else:
            outcome.append(0)
    #print("Total values predicted",len(outcome))
    #print("Total correct values predicted",sum(outcome))


    # In[63]:


    #import matplotlib.pyplot as plt
    #plt.hist(outcome)
    #plt.show()


    # In[64]:


    #len(outcome)


    # In[78]:


    #sum(outcome)


    # In[66]:

    method = "NB"
    # Actual vs predicted values
    prediction_df = pd.DataFrame()
    prediction_df['actual_id'] = pred_wordtovec_df['encoded_y']
    prediction_df['pred_id'] = pred
    prediction_df["actuals"] = actual_labels
    prediction_df["predicted"] = pred_labels
    prediction_df.to_csv('./../reports/pred_data_for_'+method+'.csv' ,index=False)
    #print("CSV with Predictions saved in following path",'./../reports/pred_data_for_'+method+'.csv')
    
    return pred , pred_labels
##
### In[67]:
##
##
##scatter_df = prediction_df.groupby(['actual_id','pred_id']).size().reset_index().rename(columns={0:'count'})
###print(scatter_df)
##plt.scatter(prediction_df['actual_id'],prediction_df['pred_id'],s=scatter_df['count'],alpha=0.5)
##
##
### In[70]:
##
##
##plot_freq(prediction_df,'pred_id')
##
##
### In[71]:
##
##
##plot_freq(prediction_df,'actual_id')
##
##
### In[ ]:
##
##
##not_most_common = prediction_df[prediction_df['actuals'] == 'Central Board of Direct Taxes (Income Tax)']
##
##
### In[ ]:
##
##
##not_most_common
##
##
### In[ ]:
##
##
###print(classification_report(y_true, y_pred))
##
##
### In[ ]:
##
##
##
##
##
### In[72]:
##
##
##from sklearn.metrics import classification_report
##y_true, y_pred = pred_wordtovec_df['encoded_y'], model.predict(pred_wordtovec_df[x_cols])
##report = classification_report(y_true, y_pred, output_dict=True)
##type(report)
##
##
### In[ ]:
##
##
##
##
##
### In[73]:
##
##
##import sklearn
##from sklearn.metrics import roc_auc_score
##y_prob = model.predict_proba(X_test)
###print(y_true.shape, y_pred.shape)
##
##print('The scikit-learn version is {}.'.format(sklearn.__version__))
##
##macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
##                                  average="macro")
##weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
##                                     average="weighted")
##macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
##                                  average="macro")
##weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
##                                     average="weighted")
##print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
##      "(weighted by prevalence)"
##      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
##print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
##      "(weighted by prevalence)"
##      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
##
##
### In[74]:
##
##
##import pandas as pd
##dict_root = dict()
##macro_roc_auc_ovo = macro_roc_auc_ovo # replace this with your variable
##weighted_roc_auc_ovo = weighted_roc_auc_ovo # replace this with your variable
##dict_root['One-vs-One'] = dict({'macro':macro_roc_auc_ovo,
##                                              'weighted_by_prevalence':weighted_roc_auc_ovo})
##macro_roc_auc_ovr = macro_roc_auc_ovr # replace this with your variable
##weighted_roc_auc_ovr = weighted_roc_auc_ovr # replace this with your variable
##dict_root['One-vs-Rest'] = dict({'macro':macro_roc_auc_ovr,
##                                              'weighted_by_prevalence':weighted_roc_auc_ovr})
##
##roc_auc_df = pd.DataFrame(dict_root)
##roc_auc_df
#### Since the row names are not the same as existig csv you are saving...
#### I suggest we put this as new csv file..like roc_report_+NAME+
##
##
### In[75]:
##
##
##type(macro_roc_auc_ovo)
##
##
### In[76]:
##
##
##for key, value in report.items():
##    print(key)
##print(report['accuracy'])
##print(report['macro avg'])
##print(report['weighted avg'])
##
##
### In[77]:
##
##
##
##saved_report= pd.DataFrame.from_dict(report)
##method = 'NB'
##cv='1111'
##filename_pred = 'prediction_report_'+method+'_'+cv+'.csv'
##filename_roc_auc = 'roc_auc_'+method+'_'+cv+'.csv'
##print(filename)
##saved_report.to_csv("./../reports/"+filename_pred)
##prediction_df.to_csv('./../reports/pred_data_for_' +method+'_'+cv+'.csv' )
##roc_auc_df.to_csv("./../reports/"+filename_roc_auc)
##
##
### In[ ]:
##
##
##
##
