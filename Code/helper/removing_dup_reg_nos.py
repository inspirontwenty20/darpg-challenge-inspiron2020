#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Functions for removing all the registrations where the department was changed. 
# TOOK the last index - assumption last is recent
# Author @inspiron
def removing_dup_reg_nos(requests_df):
    import pandas as pd
    index = requests_df.groupby('registration_no').apply(lambda x: x.index.values.min())
    remove_non_unique = requests_df.loc[index,:]
    remove_non_unique.reset_index(inplace=True, drop=True)
    #remove_non_unique.drop('index', axis=1)
    requests_df = remove_non_unique
    return requests_df

