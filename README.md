# darpg-challenge-inspiron2020
Source code for AI and ML category. 
Making the redressal process more robust and data-driven to reduce the Grievance submission and resolution lifecycle. Technology such as AI and ML could be used.

#### Pre-requisites 
Anaconda with Python version 3.7.3.
Packages used:
                              nltk,
                              sklearn,
                              matplotlib, sckit-learn

#### Getting data from API in Store_API_data_locally.ipynb :
We used the API key and accesed data for getting original request data and data for movement of requests.

#### Selecting the registration ID data using Registration_ID_Across_Files.ipynb
Use the data files saved in the Store_API_data_locally.ipynb to read the original request data and request movement data.
Select the action names from request_movement data for shortlisting the requests which have been addressed by the correct department and closed. We use the org_name to refer to the exact department name from the give data about NodalOfficers (mentioned below)
NodalOfficer_Details.xlsx is used to get the details of the departments and how they relate to org_name mentioned in the request_details.
This data is then stored in a single csv file in the Data folder locally and selected columns are used for building the model for prediciton of the departments.

#### Model_builder.ipynb :
This file has been used to train and try multiple models. It has sections defined for selection of data, defining and training models and then testing on the blind data. The first section performs the pre-processing of the text and gets a bag of words to train get the wordtovec for the given text.
When improvement is made for getting better features, the helper functions for the preprocessing will be changed.

#### Patterns_Ministry_ActionName.ipynb :
This file is used to analyse the distribution of Department names in the final data set which clearly shows that the Imbalance in the departments. We have handled this in the modelling by downsampling and using class_weight="balanced" property.
