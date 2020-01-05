#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Use this block only if you wish to view the pretty JSON.

def jsonprint(jsondata):
    import json
    # create a formatted string of the Python JSON object
    json_format = json.dumps(jsondata, sort_keys=True, indent=4)
    print(json_format)

#jsonprint(response.json())


def get_original_cpgrams(limit=1000,offset=0):
    import requests
    import pandas as pd
    #Send request with limit value to get input.
    response = requests.get("https://api.data.gov.in/restricted/cpgrams?api-key=579b464db66ec23bdd000001c3ae64f4a68d4c81770c060b13296d34&format=json&offset="+str(offset)+"&limit="+str(limit))
    print("The api request returned with resoponse code",str(response.status_code))
    data = response.json()
    #jsonprint(data)
    # "records" contains the data for grievances in the system.
    request_df = pd.DataFrame.from_records(data["records"])

    # the "desc": "msme", Do not know what this means but does not seem relevant.

    #No. of rows captured
    print("No. of rows captured",request_df.shape[0], "with offset = ",offset) # Should be equal to number of limit

    #Format is assert <condition, error message when the condition is not true>
    #assert int(limit) != (request_movement_df.shape[0]),"Value of the records inconsistent with record parameter"

    #Visualize the format of the data and see the column names.
    print(request_df.head())
    print("------------------------------------------------------------------------------------")
    print("Columns are",request_df.columns.to_list())
    return request_df

def get_movement_cpgrams(limit=1000,offset=0):
    import pandas as pd
    import requests
    #Send request with limit value to get input.
    response = requests.get("https://api.data.gov.in/restricted/movement_cpgrams?api-key=579b464db66ec23bdd000001c3ae64f4a68d4c81770c060b13296d34&format=json&offset="+str(offset)+"&limit="+str(limit))
    print("The api request returned with resoponse code",str(response.status_code))
    data = response.json()
    #jsonprint(data)
    # "records" contains the data for grievances in the system.
    request_movement_df = pd.DataFrame.from_records(data["records"])

    # the "desc": "msme", Do not know what this means but does not seem relevant.

    #No. of rows captured
    print("No. of rows captured",request_movement_df.shape[0], "from offset = ", offset) # Should be equal to number of limit

    #Format is assert <condition, error message when the condition is not true>
    #assert int(limit) != (request_movement_df.shape[0]),"Value of the records inconsistent with record parameter"

    #Visualize the format of the data and see the column names.
    print(request_movement_df.head())
    print("------------------------------------------------------------------------------------")
    print("Columns are",request_movement_df.columns.to_list())
    return request_movement_df

