#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                      'salary', 'deferred_income', 'long_term_incentive',
                      'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                      'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)

### Task 2: Remove outliers & missing values

#Remove unwanted TOTAL value
df = df.drop('TOTAL')

#Cant run algorithms with string values so replace "NaN" strings with 0 for financial features
# For email features replace with median of column
df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())

### Task 3: Create new feature(s)
df_new = df.copy()
df_new['fracion_msgs_to_poi'] = df.from_this_person_to_poi / df.from_messages
df_new['fracion_msgs_from_poi'] = df.from_poi_to_this_person / df.to_messages

data_dict = df_new.to_dict('index')

features_list = [u'poi', u'salary', u'to_messages', u'deferral_payments', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'expenses', u'from_messages', u'other',
       u'long_term_incentive', u'fracion_msgs_to_poi',
       u'fracion_msgs_from_poi']

### Extract features and labels from dataset for local testing

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
data = df_new.drop(['poi', 'email_address'], axis=1)
X = data.values
y = df.poi

# Generate the training and testing data
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train)

X_train_pca2 = pca.transform(X_train)
X_test_pca2 = pca.transform(X_test)

pca = PCA(n_components=5)
pca.fit(X_train)

X_train_pca5 = pca.transform(X_train)
X_test_pca5 = pca.transform(X_test)

from sklearn.feature_selection import SelectPercentile

X_train_reduce30 = SelectPercentile(percentile=30).fit_transform(X_train, y_train)
X_test_reduce30 = SelectPercentile(percentile=30).fit_transform(X_test, y_test)

X_train_reduce10 = SelectPercentile().fit_transform(X_train, y_train)
X_test_reduce10 = SelectPercentile().fit_transform(X_test, y_test)

## The comparison can take a while to run so uncomment if you wish to run.

# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
#
# from sklearn.cross_validation import cross_val_score, cross_val_predict
# from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
# from time import time
#
# import warnings
#
# models = {"AdaBoost Classifier": AdaBoostClassifier(), "Gaussian Naive Bayes": GaussianNB(),
#           "Decision Tree Classifier": tree.DecisionTreeClassifier(), "Random Forest Classifier": RandomForestClassifier()}
#
#
# data = {"": X_train, "(30% of features)": X_train_reduce30, "(10% of features)": X_train_reduce10,
#        "(PCA: 2 components)": X_train_pca2, "(PCA: 5 components)": X_train_pca5}
#
# res = {}
#
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#
#     for name, model in models.items():
#         for scale, X in data.items():
#             n = name + " " + scale
#             clf = model
#             accuracy = cross_val_score(clf, X_train, y_train).mean()
#
#             y_pred = cross_val_predict(clf, X_train, y_train)
#
#             rec = cross_val_score(clf, X_train, y_train, scoring=make_scorer(recall_score)).mean()
#             prec = cross_val_score(clf, X_train, y_train, scoring=make_scorer(recall_score)).mean()
#             f1 = cross_val_score(clf, X_train, y_train, scoring=make_scorer(recall_score)).mean()
#
#
#             res[n] = {"MeanAccuracy": accuracy, "Precision": prec, "Recall": rec, "F1Score": f1}
#
# results = pd.DataFrame.from_dict(res, orient="index")
# results = results[["MeanAccuracy", "Precision", "Recall", "F1Score"]]
#
# results


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, features_list)
