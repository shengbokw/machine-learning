#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

# import pickle
# import numpy as np
#
# enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#
# # print enron_data["SKILLING JEFFREY K"]["total_payments"]
# # print enron_data["FASTOW ANDREW S"]["total_payments"]
# # print enron_data["LAY KENNETH L"]["total_payments"]
#
# salary_num = 0
# address_num = 0
# for data in enron_data:
#     if np.isnan(float(enron_data[data]["total_payments"])):
#         salary_num += 1
#     address_num += 1
#
# print enron_data["LAY KENNETH L"]
#
# print salary_num, address_num


def enron_data_set():
    import numpy as np
    import pickle
    from pandas import DataFrame, Series
    import pandas as pd
    enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
    df_poi_names = pd.read_table("../final_project/poi_names.txt")

    # calculate people in the data set
    npeople = len(enron_data)
    nfeatures = len(enron_data['SKILLING JEFFREY K'])

    feature_keys = enron_data['SKILLING JEFFREY K'].keys()

    poi = []
    nsalary = 0
    nbonus = 0
    nemailadd = 0
    n_missing_totalpayments = 0
    n_nan_poi_totalpayments = 0
    for item in enron_data:
        poi_val = enron_data[item]['poi']
        salary = enron_data[item]['salary']
        bonus = enron_data[item]['bonus']
        emailadd = enron_data[item]['email_address']
        totalpayments = enron_data[item]['total_payments']
        if poi_val == True:
            poi.append(poi_val)
        if (totalpayments == "NaN" and poi_val == True):
            n_nan_poi_totalpayments += 1
        if salary != "NaN":
            nsalary += 1
        if bonus != "NaN":
            nbonus += 1
        if emailadd != "NaN":
            nemailadd += 1
        if totalpayments == "NaN":
            n_missing_totalpayments += 1

    npoi = len(poi)

    total_poi = len(df_poi_names)

    prentice = enron_data['PRENTICE JAMES']['total_stock_value']

    wc_to_poi = enron_data["COLWELL WESLEY"]['from_this_person_to_poi']

    skilling_options = enron_data['SKILLING JEFFREY K']['exercised_stock_options']

    return npeople, nfeatures, npoi, total_poi, prentice, wc_to_poi, skilling_options, nemailadd, nsalary, nbonus, n_missing_totalpayments, n_nan_poi_totalpayments


print enron_data_set()
