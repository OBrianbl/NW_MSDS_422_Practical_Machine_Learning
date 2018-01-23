# Brandon OBriant
# PREDICT 422
# Assignment 02
# Bank Telephone Direct Marketing Campaign: Classifier comparison

# prepare for Python version 3x features and functions


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# import base packages into the namespace for this program
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score   
from sklearn.model_selection import KFold

# function, takes in desired working directory path, changes the working directory
# to that path, and prints out the current working directory as a sanity check
def change_working_dir(path):
    os.chdir(path)
    print("Current working directory:{}".format(str(os.getcwd())))

# gets assigned desired working directory--put your working directory here:***
WORKING_DIRECTORY_PATH = "../MSDS_422_Bank_Marketing_Classifier_Comparison/"

# changes working dir to desire call    
change_working_dir(WORKING_DIRECTORY_PATH)

# function to load csv file into a DataFrame
def load_csv(filename):
    data = pd.read_csv(filename, sep = ';')
    return data

# prints shape of data, if there is no name associated with data (i.e. np.arrar)
# then it ommit the associate name
def print_shape(data, name = None):
    if name != None:
        data.name = name
        print("The shape of data, {}, is: {}".format(name, str(data.shape)))
    else:
        print("The shape of data is: {}".format(str(data.shape)))
    
    
# drop observations with missing data, if any
# examine the shape of input data after dropping missing data    
def dropna_print_shape(dataframe, name):
    if dataframe.isnull().values.any() == True:
        dataframe = dataframe.dropna()
        print("\n-----Dropped NAN values------\n")
        print_shape(dataframe, name)
    else:
        print("\n-----No NAN values------\n")
        print_shape(dataframe, name)
    return dataframe


# initial work with the smaller data set
bank = load_csv('bank.csv') 
# examine the shape of original input data
print_shape(bank, 'bank')

# drop observations from bank DataFrame with missing data, if any
# examine the shape of input data after dropping missing data
bank = dropna_print_shape(bank, 'bank')


# prints information statistics and descriptions about data passed in and 
# saves it to a txt file for review
def print_data_info_save_to_file(data, dataname):
    print('\n---------{} data informations----------\n'.format(dataname))
    print('\n{} data shape: {}'.format(dataname, data.shape))
    print('\n{} data column values: {}'.format(dataname, data.columns.values)) 
    print('\n{} data first few rows: {}'.format(dataname, data.head())) 
    print('\n{} data look at end of data: {}'.format(dataname, data.tail()))
    print('\n{} data descriptive statistics: {}'.format(dataname, data.describe()))
    with open("{}_data_descriptive_information.txt".format(dataname), "w") as text_file:
        text_file.write('\n---------{} data informations----------\n'.format(dataname)+
                        '\n{} data shape: {}'.format(dataname, str(data.shape)) +
                        '\n{} data column values: {}'.format(dataname, str(data.columns.values)) + 
                        '\n{} data first few rows: {}'.format(dataname, str(data.head()))+ 
                        '\n{} data look at end of data: {}'.format(dataname, str(data.tail()))+
                        '\n{} data descriptive statistics: {}'.format(dataname, str(data.describe()))+ 
                        '\n{} data information: {}'.format(dataname, str(data.info())))

# look at the list of column names, note that y is the response
# look at the beginning of the DataFrame
# Look at the end of the DataFrame
# bank descriptive statistics
print_data_info_save_to_file(bank, 'bank')



# mapping function to convert text no/yes to integer 0/1
def map_to_binary(dataframe, feature):
    mapped_df = pd.DataFrame()
    impute_to_binary = {'no' : 0, 'yes' : 1}
    mapped_df = dataframe[feature].map(impute_to_binary)
    return mapped_df
    

# define binary variable for having credit in default
default = map_to_binary(bank, 'default')

# define binary variable for having a mortgage or housing loan
housing = map_to_binary(bank, 'housing')

# define binary variable for having a personal loan
loan = map_to_binary(bank, 'loan')

# define response variable to use in the model
response = map_to_binary(bank, 'response')



# gather three explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan), 
    np.array(response)]).T

# examine the shape of model_data, which we will use in subsequent modeling
print_shape(model_data)

# prints statistics about data passed in and save it to a txt file for review
def print_stats_save_to_file(data, dataname):
    print('\n---------{} data statistics----------\n'.format(dataname))
    print('\n{} data shape: {}\n'.format(dataname, data.shape))
    print('\n{} data mean: {}\n'.format(dataname, np.mean(data)))
    print('\n{} data standard deviation: {}\n'.format(dataname, np.std(data)))
    print('\n{} data standard median: {}\n'.format(dataname, np.median(data)))
    print('\n{} data variance: {}\n'.format(dataname, np.var(data)))
    with open("model_data_descriptive_stats.txt", "w") as text_file:
        text_file.write('\n---------{} data statistics----------\n'.format(dataname)+
                        '\n{} data mean: {}'.format(dataname, str(np.mean(data))) + 
                        '\n{} data standard deviation: {}'.format(dataname, str(np.std(data)))+ 
                        '\n{} data median: {}'.format(dataname, str(np.median(data)))+
                        '\n{} data variance: {}'.format(dataname, str(np.var(data)))+ 
                        '\n{} data shape: {}'.format(dataname, str(data.shape)))

# prints model_data statistics and saves it to a txt file    
print_stats_save_to_file(model_data, 'Model')    
    

# shuffle the rows 
np.random.seed(RANDOM_SEED)
np.random.shuffle(model_data)

# examine the shape of model_data, after shuffle, which we will use in subsequent modeling
print_shape(model_data)

# list of names for classifier models
classifier_names = ["Logistic_Regression", "Naive_Bayes"]

# list of classifiers
classifiers = [LogisticRegression(), BernoulliNB(alpha=1.0, binarize=0.5, 
                           class_prior = [0.5, 0.5], fit_prior=False)]


# ten-fold cross-validation employed here
N_FOLDS = 10

# set up numpy array for storing results
crossvalidation_results = np.zeros((N_FOLDS, len(classifier_names)))

# kf, object,  model selection kfold split set up
kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

#--check the splitting process by looking at fold observation counts--
# fold count initialized to zero
index_for_fold = 0 

# splits the data, fits the classifier models, returns the crossvalidation
# results
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    # 0:model_data.shape[1]-1 slices for explanatory variables,
    X_train = model_data[train_index, 0:model_data.shape[1]-1]
    X_test = model_data[test_index, 0:model_data.shape[1]-1]
    
    # model_data.shape[1]-1 is the index for the response variable
    y_train = model_data[train_index, model_data.shape[1]-1]
    y_test = model_data[test_index, model_data.shape[1]-1]
    
    # prints structure of data after split for x, y 
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)
    
    # index for method initialized to zero
    index_for_method = 0
    
    # loops through classifiers
    # fits the respective model
    # performs predictions
    for name, clf in zip(classifier_names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        
         # fit current classifier model using train data set
        clf.fit(X_train, y_train) 
        
        # calculate predictions to evaluate, using test set for this fold
        y_test_predict = clf.predict_proba(X_test)
        
        # calculates ROC AUC score, stores results in cv_results
        fold_method_result = roc_auc_score(y_test, y_test_predict[:,1]) 
        print('Area under ROC curve:', fold_method_result)
        crossvalidation_results[index_for_fold, index_for_method] = fold_method_result
        
        
        # adds one to the index, next loop will be the next classifier
        index_for_method += 1
        
    # adds one to the index, next loop will be the next split in fold    
    index_for_fold += 1

# pandas DataFrame gets assigned cross fold evaluation results
crossvalidation_results_df = pd.DataFrame(crossvalidation_results)
crossvalidation_results_df.columns = classifier_names
with open("cv-results-df.txt", "w") as text_file:
    text_file.write('\nResults from '+ str(N_FOLDS) + '-fold cross-validation\n'+
                     '\nMethod Area under ROC Curve:\n'+ 
                     str(crossvalidation_results_df))

# print mean of ROC AUC evaluation results for each classifier, saves to file
print('\n----------------------------------------------')
print('\nAverage results from {}-fold cross-validation\n\nMethod Area under ROC Curve:\n{}'
      .format(str(N_FOLDS),str(crossvalidation_results_df.mean())), sep = '')     
print('\nMean of cross validation result: {}'.format(crossvalidation_results_df.mean())) 
with open("cv-results-df-mean.txt", "w") as text_file:
    text_file.write('\nAverage results from {}-fold cross-validation\n\nMethod Area under ROC Curve:\n{}'
                    .format(str(N_FOLDS),str(crossvalidation_results_df.mean())))




    















