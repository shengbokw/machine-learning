#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest

def remove_outliers(dictionary, outliers):
    """ this function removes a list of keys from a dictionary object """
    for outlier in outliers:
        dictionary.pop(outlier, 0)

def get_nan_counts(dictionary):
    '''
    converts 'NaN' string to np.nan returning a pandas
    dataframe of each feature and it's corresponding
    percent null values (nan)
    '''
    my_df = pd.DataFrame(dictionary).transpose()
    nan_counts_dict = {}
    for column in my_df.columns:
        my_df[column] = my_df[column].replace('NaN', np.nan)
        nan_counts = my_df[column].isnull().sum()
        nan_counts_dict[column] = round(float(nan_counts) / float(len(my_df[column])) * 100, 1)
    df = pd.DataFrame(nan_counts_dict, index=['percent_nan']).transpose()
    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'index': 'feature'})
    return df

def get_k_best(dictionary, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection returning:
    {feature:score}
    """
    data = featureFormat(dictionary, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    pairs = zip(features_list[1:], scores)
    # combined scores and features into a pandas dataframe then sort
    k_best_features = pd.DataFrame(pairs, columns=['feature', 'score'])
    k_best_features = k_best_features.sort('score', ascending=False)

    # merge with null counts
    df_nan_counts = get_nan_counts(dictionary)
    k_best_features = pd.merge(k_best_features, df_nan_counts, on='feature')

    # eliminate infinite values
    k_best_features = k_best_features[np.isinf(k_best_features.score) == False]
    print 'Feature Selection by k_best_features\n'
    print "{0} best features in descending order: {1}\n".format(k, k_best_features.feature.values[:k])
    print '{0}\n'.format(k_best_features[:k])

    return k_best_features[:k]


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'

email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]

financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
    ]

features_list = [target_label] + financial_features_list + email_features_list



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']

remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)
def compute_fraction(poi_messages, all_messages):
    """ return fraction of messages from/to that person to/from POI"""
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.
    fraction = poi_messages / all_messages
    return fraction


for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

my_feature_list = features_list + ['fraction_from_poi', 'fraction_to_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

best_features = get_k_best(my_dataset, my_feature_list, 10)

### get the 10 best features are ['exercised_stock_options' 'total_stock_value' 'bonus' 'salary'
### 'deferred_income' 'long_term_incentive' 'restricted_stock'
### 'total_payments' 'shared_receipt_with_poi' 'loan_advances']

final_feature_list = [target_label] + ['exercised_stock_options', 'total_stock_value', 'bonus', 'salary',
                                         'deferred_income', 'long_term_incentive', 'restricted_stock',
                                         'total_payments', 'shared_receipt_with_poi', 'loan_advances']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, final_feature_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##########################Task 4: Using algorithm########################
# 4.0 scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
g_clf = GaussianNB()

###4.1  Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

###4.2  K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)


###4.3 Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000, gamma=0.0001, random_state=42, class_weight='auto')

###4.4 Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth=5, max_features='sqrt', n_estimators=10, random_state=42)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
###5.1 evaluate function
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import mean

def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "done.\n"
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)


### 5.2 Evaluate all functions
evaluate_clf(g_clf, features, labels)
evaluate_clf(l_clf, features, labels)
evaluate_clf(k_clf, features, labels)
evaluate_clf(s_clf, features, labels)
evaluate_clf(rf_clf, features, labels)

### Select Logistic Regression as final algorithm
clf = l_clf


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
# dump your classifier, dataset and features_list so
# anyone can run/check your results

pickle.dump(clf, open("./my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("./my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("./my_feature_list.pkl", "w"))