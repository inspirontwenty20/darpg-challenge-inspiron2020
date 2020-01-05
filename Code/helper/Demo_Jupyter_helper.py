#!/usr/bin/env python
# coding: utf-8


import helper_import_path
import model_executor_for_demo as mba

#converting it into a data frame:
import pandas as pd
import numpy as np
def get_prediction_on_input_data(model,encoder_array,country_name, state_name, subject_content):
    test_df = pd.DataFrame(np.nan, index = np.arange(1), columns = ['registration_no',
                                                                   'encoded_y',
                                                                   'country_name',
                                                                   'state_name',
                                                                   'subject_content',
                                                                   'Apex_org_name'])
    actual_class = 'Department of Health & Family Welfare'
    encoded_y = 0
    
    test_df.loc[0,:] = ['AYUSH/E/2019/00606',encoded_y,country_name, state_name, subject_content, \
                        actual_class] 
    y_colname = 'Apex_org_name'
    selected_columns = ['registration_no','encoded_y', 'country_name','state_name', 'subject_content',y_colname]
    test_reg_list = list(test_df['registration_no'])
    #print("Test dataframe", test_df)
    pred, pred_labels = mba.predict_on_test_data(model,encoder_array,selected_columns,test_df,test_reg_list)
    return test_df,pred, pred_labels


def read_run_from_file():
    import pandas as pd
    test_file_name = './../reports/test_data_for_prediction.csv'
    test_df = pd.read_csv(test_file_name)
    y_colname = 'Apex_org_name'
    selected_columns = ['registration_no','encoded_y', 'country_name','state_name', 'subject_content',y_colname]
    test_reg_list = list(test_df['registration_no'])
    mba.predict_on_test_data(model,encoder_array,selected_columns,test_df,test_reg_list)
    

