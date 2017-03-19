from mq_mergedata import *
from mq_dataprocessing import *
from mq_modelbuilder import *
from sklearn.cross_validation import train_test_split

# # Load Data
print 'Load Data'
merged_table = 'train_All2.csv'
df = pd.read_csv(merged_table)
#df = df.sample(frac=0.15, replace=False)  # For testing !!
kaggle_table = 'test_All2.csv'
kaggle_test = pd.read_csv(kaggle_table)
#kaggle_test = kaggle_test.sample(frac=0.01, replace=False)  # For testing !!!

# Feature Generation
# Generate Features
print 'Generate data'
df1 = feature_generation(df)
kaggle_test1 = test_feature_generation(kaggle_test)
del df

# # Clean Data
print 'Clean Data'
X_clf = preprocessing_data(df1)
kaggle_clf = test_preprocessing_data(kaggle_test1)
del df1, kaggle_test1

# # Split train and test
print 'Split Data'
train, test = train_test_split(X_clf, test_size=0.20, random_state=27)

X_train, y_train, X_test, y_test = train, train.pop('is_install'), test, test.pop('is_install')
del train, test

# Feature Hashing
print('Feature Creating/Hashing Train')
feature_creator = FeatureCreator()
design_matrix_transformer = FeatureHasher(
    22, cat_features_pc, None, cat_interactions_pc, store_fmap=True)

X_test = feature_creator.transform(X_test, inplace=True)
X_test, f_map = design_matrix_transformer.fit_transform(X_test)

X_train = feature_creator.transform(X_train, inplace=True)
X_train, f_map = design_matrix_transformer.fit_transform(X_train)

# Modeling
print 'Modeling'
logistic_baseline = logistic_model(X_train, y_train)

# Calculate prediction/probability of train and test
X_train_predictions = logistic_baseline.predict(X_train)
X_train_predprob = logistic_baseline.predict_proba(X_train)[:, 1]

X_test_predictions = logistic_baseline.predict(X_test)
X_test_predprob = logistic_baseline.predict_proba(X_test)[:, 1]

# Calculate metrics of train, validation and test set.
# lr_ll_val = -logistic_baseline.best_score_

lr_ll_train = log_loss(y_train, X_train_predprob)
lr_auc_train = roc_auc_score(y_train, X_train_predprob)

lr_ll_test = log_loss(y_test, X_test_predprob)
lr_auc_test = roc_auc_score(y_test, X_test_predprob)

# Print out the results

# print "Best parameter: ", logistic_baseline.best_params_
# print "Log Loss (Validation): %f" % lr_ll_val

print "Log Loss (Train): %f" % lr_ll_train
print "AUC (Train): %f" % lr_auc_train

print 'Log Loss (Test): %f' % lr_ll_test
print 'AUC (Test): %f' % lr_auc_test

# Predict Kaggle Test dataset
print('Feature Creating/Hashing Kaggle Test data')
feature_creator = FeatureCreator()
design_matrix_transformer = FeatureHasher(
    22, cat_features_pc, None, cat_interactions_pc, store_fmap=True)

kaggle_clf = feature_creator.transform(kaggle_clf, inplace=True)
kaggle_clf, f_map = design_matrix_transformer.fit_transform(kaggle_clf)

# Predict
print 'Predict Kaggle Testset'
K_test_predprob = logistic_baseline.predict_proba(kaggle_clf)[:, 1]
kaggle_test['y'] = K_test_predprob
output = kaggle_test[['row', 'y']]
# write your pandas table into files on S3
UPLOAD_NAME = 'predicted20170305_pycharm2.csv'
output.to_csv(UPLOAD_NAME, index=False)








