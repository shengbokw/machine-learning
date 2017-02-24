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

import pickle
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print enron_data["SKILLING JEFFREY K"]["total_payments"]
# print enron_data["FASTOW ANDREW S"]["total_payments"]
# print enron_data["LAY KENNETH L"]["total_payments"]

salary_num = 0
address_num = 0
for data in enron_data:
    if np.isnan(float(enron_data[data]["total_payments"])):
        salary_num += 1
    address_num += 1

print enron_data["LAY KENNETH L"]

print salary_num, address_num
